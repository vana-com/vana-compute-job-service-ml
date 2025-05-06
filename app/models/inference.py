import time
from pathlib import Path
from typing import Dict, Any, Generator, Union
import torch
import json

# Import Unsloth for efficient inference
from unsloth import FastLanguageModel
from transformers import TextIteratorStreamer
from threading import Thread

from app.config import settings

# Cache for loaded models to avoid reloading
model_cache = {}

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
        model_path=str(model_path),
        max_seq_length=settings.MAX_SEQ_LENGTH,
        dtype=torch.bfloat16,
        load_in_4bit=True
    )
    
    # Cache the model
    model_cache[cache_key] = (model, tokenizer)
    
    return model, tokenizer

def generate_text(
    model_path: Path,
    prompt: str,
    max_new_tokens: int = settings.MAX_NEW_TOKENS,
    temperature: float = settings.TEMPERATURE,
    top_p: float = settings.TOP_P,
    stream: bool = False
) -> Union[Dict[str, Any], Generator[str, None, None]]:
    """
    Generate text using the specified model.
    
    Args:
        model_path: Path to the model directory
        prompt: Input prompt for text generation
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        stream: Whether to stream the output
        
    Returns:
        If stream=False, returns a dictionary with the generated text and metadata.
        If stream=True, returns a generator that yields chunks of text.
    """
    # Load model
    model, tokenizer = load_model(model_path)
    
    # Format prompt if needed
    if not prompt.startswith("<s>[INST]"):
        prompt = f"<s>[INST] {prompt} [/INST]"
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Set up generation config
    generation_config = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
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
        
        # Stream the output
        def generate_stream():
            for text in streamer:
                yield f"data: {json.dumps({'text': text})}\n\n"
            yield "data: [DONE]\n\n"
        
        return generate_stream()
    
    # Non-streaming generation
    start_time = time.time()
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            **generation_config
        )
    
    # Decode the output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the prompt from the generated text
    prompt_decoded = tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)
    if generated_text.startswith(prompt_decoded):
        generated_text = generated_text[len(prompt_decoded):].strip()
    
    generation_time = time.time() - start_time
    
    return {
        "text": generated_text,
        "generation_time": generation_time
    }