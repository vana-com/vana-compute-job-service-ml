import time
from pathlib import Path
from typing import Generator, Union, List, Optional
import torch
import tiktoken
from utils.devices import supported_dtype

# Import Unsloth for efficient inference
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextIteratorStreamer
from threading import Thread

from config import settings
from models.schemas import (
    Message, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice, 
    ChatCompletionResponseUsage, 
    ChatCompletionChunk, 
    ChatCompletionChunkChoice
)

# Cache for loaded models to avoid reloading
model_cache = {}
# Cache for tokenizers to count tokens
tokenizer_cache = {}

def load_model(model_path: Path):
    """
    Load a model from the specified path.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        tuple: (model, tokenizer)
    """
    # Check if model is already loaded
    cache_key = str(model_path)
    if cache_key in model_cache:
        return model_cache[cache_key]
    
    # Load model with Unsloth
    model, tokenizer = FastLanguageModel.from_pretrained(
        str(model_path),
        max_seq_length=settings.MAX_SEQ_LENGTH,
        dtype=supported_dtype,
        load_in_4bit=True
    )
    
    tokenizer = get_chat_template(tokenizer, chat_template="alpaca")

    # Enable inference kernels
    FastLanguageModel.for_inference(model)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Cache the model
    model_cache[cache_key] = (model, tokenizer)
    
    return model, tokenizer

def count_tokens(text: str, model_name: str = "gpt-3.5-turbo") -> int:
    """
    Count the number of tokens in a text string.
    
    Args:
        text: The text to count tokens for
        model_name: The model to use for counting tokens
        
    Returns:
        int: The number of tokens
    """
    if model_name not in tokenizer_cache:
        try:
            tokenizer_cache[model_name] = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            tokenizer_cache[model_name] = tiktoken.get_encoding("cl100k_base")
    
    tokenizer = tokenizer_cache[model_name]
    return len(tokenizer.encode(text))

def format_prompt_from_messages(messages: List[Message], tokenizer=None) -> str:
    """
    Format a prompt from a list of messages in the OpenAI format.
    
    Args:
        messages: List of messages in the OpenAI format
        tokenizer: Optional tokenizer to use its chat template
        
    Returns:
        str: Formatted prompt for the model
    """
    # If we have a tokenizer, use its built-in chat template
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        # Convert our Message objects to the format expected by apply_chat_template
        formatted_messages = []
        for msg in messages:
            formatted_messages.append({"role": msg.role, "content": msg.content})
        
        return tokenizer.apply_chat_template(
            formatted_messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
    # Fallback to manual formatting
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
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Format prompt from messages using tokenizer's chat template
    prompt = format_prompt_from_messages(messages, tokenizer)
    
    # Tokenize input
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=settings.MAX_SEQ_LENGTH,
        return_attention_mask=True
    ).to(model.device)
    
    # Count tokens
    prompt_tokens = inputs.input_ids.shape[1]
    
    # Set up generation config
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
        import uuid
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    
    # Handle streaming
    if stream:
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
        
        # Stream the output in OpenAI format
        async def generate_stream():
            content_so_far = ""
            
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
            for text in streamer:
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
        
        return generate_stream()
    
    # Non-streaming generation
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            **generation_config
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    prompt_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    if generated_text.startswith(prompt_decoded):
        generated_text = generated_text[len(prompt_decoded):].strip()
    
    # Count completion tokens
    completion_tokens = count_tokens(generated_text)
    total_tokens = prompt_tokens + completion_tokens
    
    # Create the response
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
    
    return response