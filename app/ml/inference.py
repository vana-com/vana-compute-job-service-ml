import time
import logging
import traceback
import os
import sys
import psutil
from pathlib import Path
from typing import Generator, Union, List, Optional
import torch
import tiktoken
import uuid
from app.utils.devices import supported_dtype

from unsloth import FastLanguageModel, __version__ as unsloth_version
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer, __version__ as transformers_version
from threading import Thread

from app.config import settings
from app.models.openai import (
    Message, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice, 
    ChatCompletionResponseUsage, 
    ChatCompletionChunk, 
    ChatCompletionChunkChoice
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Cache for loaded models to avoid reloading
model_cache = {}
# Cache for tokenizers to count tokens
tokenizer_cache = {}

def log_system_info():
    """Log system and library information for debugging"""
    logger.info(f"System information:")
    logger.info(f"- Python: {sys.version}")
    logger.info(f"- PyTorch: {torch.__version__}")
    logger.info(f"- Unsloth: {unsloth_version}")
    logger.info(f"- Transformers: {transformers_version}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        try:
            logger.info(f"Device capabilities: {torch.cuda.get_device_capability(0)}")
            logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
        except Exception as e:
            logger.warning(f"Could not get detailed GPU info: {str(e)}")
    else:
        logger.warning("No GPU available. CPU-only inference will be slow.")
    
    # Log memory information
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    logger.info(f"Process memory: {memory_info.rss / 1024 / 1024:.2f} MB")

def load_model(model_path: Path):
    """
    Load a model from the specified path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    start_time = time.time()
    
    try:
        # Check if model is already loaded
        cache_key = str(model_path)
        if cache_key in model_cache:
            logger.info(f"Using cached model: {cache_key}")
            return model_cache[cache_key]
        
        logger.info(f"Loading model from: {model_path}")
        
        if not os.path.exists(model_path):
            error_msg = f"Model path does not exist: {model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        # Log memory before loading
        if torch.cuda.is_available():
            before_gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            logger.info(f"GPU memory allocated before loading: {before_gpu_allocated:.2f} MB")
        
        # Load model with Unsloth
        logger.info(f"Loading model with max_seq_length={settings.MAX_SEQ_LENGTH}, dtype={supported_dtype}")
        
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                str(model_path),
                max_seq_length=settings.MAX_SEQ_LENGTH,
                dtype=supported_dtype,
                load_in_4bit=True
            )
        except Exception as e:
            error_msg = f"Failed to load model from {model_path}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
        
        logger.info(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
        
        try:
            tokenizer = get_chat_template(tokenizer, chat_template="alpaca")
            logger.info(f"Applied 'alpaca' chat template to tokenizer")
        except Exception as e:
            error_msg = f"Failed to apply chat template: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e

        # Enable inference kernels
        try:
            FastLanguageModel.for_inference(model)
            logger.info("Enabled inference kernels for faster generation")
        except Exception as e:
            logger.warning(f"Failed to enable inference kernels: {str(e)}. Will continue with standard generation.")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token because pad_token was None")
        
        # Log memory after loading
        if torch.cuda.is_available():
            after_gpu_allocated = torch.cuda.memory_allocated() / 1024 / 1024
            logger.info(f"GPU memory allocated after loading: {after_gpu_allocated:.2f} MB")
            logger.info(f"GPU memory used by model: {after_gpu_allocated - before_gpu_allocated:.2f} MB")
        
        # Log model info
        if hasattr(model, 'get_memory_footprint'):
            try:
                model_size_gb = model.get_memory_footprint() / 1024 / 1024 / 1024
                logger.info(f"Model memory footprint: {model_size_gb:.2f} GB")
            except Exception as e:
                logger.warning(f"Could not get model memory footprint: {str(e)}")
        
        # Cache the model
        model_cache[cache_key] = (model, tokenizer)
        logger.info(f"Model cached with key: {cache_key}")
        
        return model, tokenizer
        
    except Exception as e:
        error_msg = f"Error loading model from {model_path}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg) from e

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model_name: The model to use for counting tokens
        
    Returns:
        int: The number of tokens
    """
    try:
        if model_name not in tokenizer_cache:
            logger.debug(f"Creating tokenizer for {model_name}")
            try:
                tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
                logger.debug(f"Using tiktoken encoding for {model_name}")
            except KeyError:
                # Fallback to cl100k_base encoding if model not found
                logger.warning(f"Model {model_name} not found in tiktoken, falling back to cl100k_base")
                tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
        
        tokenizer = tokenizer_cache[model_name]
        token_count = len(tokenizer.encode(text))
        logger.debug(f"Text contains {token_count} tokens with {model_name} tokenizer")
        return token_count
    except Exception as e:
        logger.error(f"Error counting tokens: {str(e)}")
        logger.error(traceback.format_exc())
        # Return an estimated count to avoid breaking the calling code
        return len(text) // 4  # Very rough estimate

def format_prompt_from_messages(messages: List[Message], tokenizer=None) -> str:
    """
    Format a prompt from a list of messages in the OpenAI format.
    
    Args:
        messages: List of messages in the OpenAI format
        tokenizer: Optional tokenizer to use its chat template
        
    Returns:
        str: Formatted prompt for the model
    """
    try:
        # If we have a tokenizer, use its built-in chat template
        if tokenizer and hasattr(tokenizer, "apply_chat_template"):
            # Convert our Message objects to the format expected by apply_chat_template
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({"role": msg.role, "content": msg.content})
            
            logger.debug(f"Using tokenizer's built-in chat template")
            result = tokenizer.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return result
        
        # Fallback to manual formatting
        logger.debug(f"Using manual chat template formatting (tokenizer.apply_chat_template not available)")
        prompt = ""
        
        for message in messages:
            if message.role == "system":
                # Add system message at the beginning
                prompt += f"<s>[INST] <<SYS>>\n{message.content}\n<</SYS>>\n\n"
            elif message.role == "user":
                # If we already have a system message, just add the user message
                if prompt:
                    prompt += f"{message.content} [/INST]"
                else:
                    # Otherwise, start a new instruction
                    prompt += f"<s>[INST] {message.content} [/INST]"
            elif message.role == "assistant":
                # Add assistant response
                prompt += f" {message.content} </s>"
                # Start a new instruction if there are more messages
                if message != messages[-1]:
                    prompt += f"<s>[INST] "
        
        # If the last message is from the user, we need to close the instruction
        if messages[-1].role == "user" and not prompt.endswith("[/INST]"):
            prompt += " [/INST]"
        
        return prompt
    except Exception as e:
        error_msg = f"Error formatting prompt from messages: {str(e)}"
        logger.error(error_msg)
        logger.error(f"Messages: {messages}")
        logger.error(traceback.format_exc())
        raise RuntimeError(error_msg) from e

async def generate_chat_completion(
    model_path: Path,
    messages: List[Message],
    temperature: float = settings.TEMPERATURE,
    top_p: float = settings.TOP_P,
    max_tokens: int = settings.MAX_NEW_TOKENS,
    stop: Optional[Union[str, List[str]]] = None,
    stream: bool = False,
    completion_id: str = None,
    created: int = None
) -> Union[ChatCompletionResponse, Generator[str, None, None]]:
    """
    Generate a chat completion following the OpenAI API specification.
    
    Args:
        model_path: Path to the model directory
        messages: List of messages in the OpenAI format
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum number of tokens to generate
        stop: Stop sequences
        stream: Whether to stream the output
        completion_id: ID for the completion
        created: Timestamp for the completion
        
    Returns:
        If stream=False, returns a ChatCompletionResponse.
        If stream=True, returns a generator that yields SSE formatted chunks.
    """
    start_time = time.time()
    logger.info(f"Starting chat completion with model: {model_path.name}")
    logger.debug(f"Parameters: temp={temperature}, top_p={top_p}, max_tokens={max_tokens}, stream={stream}")
    
    # Log a sample of the input messages (truncated for privacy/brevity)
    if len(messages) > 0:
        sample_message = messages[-1]
        content_preview = sample_message.content[:50] + "..." if len(sample_message.content) > 50 else sample_message.content
        logger.debug(f"Last message: role={sample_message.role}, content preview: {content_preview}")
    
    try:
        # Load model
        try:
            model, tokenizer = load_model(model_path)
        except Exception as e:
            error_msg = f"Failed to load model at {model_path}: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Format prompt from messages using tokenizer's chat template
        try:
            prompt = format_prompt_from_messages(messages, tokenizer)
            logger.debug(f"Formatted prompt length: {len(prompt)} characters")
        except Exception as e:
            error_msg = f"Failed to format prompt: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Tokenize input
        logger.debug("Tokenizing input prompt")
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.MAX_SEQ_LENGTH,
                return_attention_mask=True
            ).to(model.device)
            logger.debug(f"Input shape: {inputs.input_ids.shape}")
        except Exception as e:
            error_msg = f"Failed to tokenize input: {str(e)}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Count tokens
        prompt_tokens = inputs.input_ids.shape[1]
        logger.debug(f"Prompt tokens: {prompt_tokens}")
        
        # Check if we have enough context length for generation
        if prompt_tokens >= settings.MAX_SEQ_LENGTH - max_tokens:
            logger.warning(f"Prompt uses {prompt_tokens} tokens, which leaves only {settings.MAX_SEQ_LENGTH - prompt_tokens} tokens for generation (requested {max_tokens})")
        
        # Set up generation config
        logger.debug("Setting up generation config")
        generation_config = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.eos_token_id,
        }
        
        # Add stop sequences if provided
        if stop:
            if isinstance(stop, str):
                stop = [stop]
            logger.debug(f"Using stop sequences: {stop}")
            generation_config["stopping_criteria"] = [
                lambda input_ids, scores: any(
                    tokenizer.decode(input_ids[0][-len(tokenizer.encode(s)):]) == s 
                    for s in stop
                )
            ]
        
        # Use current time if not provided
        if created is None:
            created = int(time.time())
        
        # Generate a unique ID if not provided
        if completion_id is None:
            completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        
        logger.debug(f"Completion ID: {completion_id}")
        
        # Handle streaming
        if stream:
            logger.debug("Using streaming mode")
            try:
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
                generation_config["streamer"] = streamer
                
                # Start generation in a separate thread
                thread = Thread(
                    target=model.generate,
                    kwargs={
                        "input_ids": inputs.input_ids,
                        **generation_config
                    }
                )
                thread.start()
                logger.debug("Started generation thread for streaming")
                
                # Stream the output in OpenAI format
                async def generate_stream():
                    content_so_far = ""
                    stream_start_time = time.time()
                    
                    # Send the first chunk with role
                    first_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_path.name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={"role": "assistant"},
                                finish_reason=None
                            )
                        ]
                    )
                    yield f"data: {first_chunk.json()}\n\n"
                    
                    # Stream the content
                    chunks_emitted = 0
                    try:
                        for text in streamer:
                            chunks_emitted += 1
                            content_so_far += text
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=model_path.name,
                                choices=[
                                    ChatCompletionChunkChoice(
                                        index=0,
                                        delta={"content": text},
                                        finish_reason=None
                                    )
                                ]
                            )
                            yield f"data: {chunk.json()}\n\n"
                    except Exception as e:
                        logger.error(f"Error during streaming: {str(e)}")
                        logger.error(traceback.format_exc())
                    
                    # Send the final chunk
                    final_chunk = ChatCompletionChunk(
                        id=completion_id,
                        created=created,
                        model=model_path.name,
                        choices=[
                            ChatCompletionChunkChoice(
                                index=0,
                                delta={},
                                finish_reason="stop"
                            )
                        ]
                    )
                    yield f"data: {final_chunk.json()}\n\n"
                    yield "data: [DONE]\n\n"
                    
                    stream_duration = time.time() - stream_start_time
                    logger.info(f"Streaming completed: generated {len(content_so_far)} chars in {stream_duration:.2f}s ({chunks_emitted} chunks)")
                
                return generate_stream()
            except Exception as e:
                error_msg = f"Error setting up streaming: {str(e)}"
                logger.error(error_msg)
                logger.error(traceback.format_exc())
                raise RuntimeError(error_msg) from e
        
        # Non-streaming generation
        logger.debug("Using non-streaming mode")
        start_time = time.time()
        
        # Generate text
        try:
            logger.debug("Starting text generation")
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    **generation_config
                )
            generation_time = time.time() - start_time
            logger.debug(f"Generation completed in {generation_time:.2f} seconds")
        except Exception as e:
            error_msg = f"Error during generation: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
        
        # Decode the output
        try:
            logger.debug("Decoding generated text")
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the prompt from the generated text
            prompt_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
            if generated_text.startswith(prompt_decoded):
                generated_text = generated_text[len(prompt_decoded):].strip()
                logger.debug("Removed prompt from generated text")
            
            logger.debug(f"Generated text length: {len(generated_text)} chars")
        except Exception as e:
            error_msg = f"Error decoding generated text: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
        
        # Count completion tokens
        try:
            completion_tokens = count_tokens(generated_text)
            total_tokens = prompt_tokens + completion_tokens
            logger.debug(f"Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
        except Exception as e:
            logger.warning(f"Error counting tokens: {str(e)}, using estimates")
            completion_tokens = len(generated_text) // 4  # Rough estimate
            total_tokens = prompt_tokens + completion_tokens
        
        # Create the response
        try:
            response = ChatCompletionResponse(
                id=completion_id,
                created=created,
                model=model_path.name,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=Message(
                            role="assistant",
                            content=generated_text
                        ),
                        finish_reason="stop"
                    )
                ],
                usage=ChatCompletionResponseUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens
                )
            )
        except Exception as e:
            error_msg = f"Error creating response object: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            raise RuntimeError(error_msg) from e
        
        total_time = time.time() - start_time
        logger.info(f"Chat completion finished in {total_time:.2f}s: {completion_tokens} tokens generated")
        
        return response
    except Exception as e:
        logger.error(f"Chat completion failed: {str(e)}")
        logger.error(traceback.format_exc())               
        # Re-raise with a more informative message
        raise RuntimeError(f"Failed to generate chat completion: {str(e)}") from e