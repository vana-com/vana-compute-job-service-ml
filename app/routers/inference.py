from fastapi import APIRouter, HTTPException, status, Request, Depends
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List, Optional, Dict, Any, Union
import os
import time
import json
import asyncio
from pathlib import Path
import uuid

from app.config import settings
from app.models.inference import generate_chat_completion
from app.models.schemas import (
    Message, 
    ChatCompletionRequest, 
    ChatCompletionResponse, 
    ChatCompletionResponseChoice,
    ChatCompletionResponseUsage,
    ChatCompletionChunk,
    ChatCompletionChunkChoice
)

router = APIRouter()

@router.post("/chat/completions", response_model=Union[ChatCompletionResponse, StreamingResponse])
async def create_chat_completion(request: ChatCompletionRequest):
    """
    Create a chat completion following the OpenAI API specification.
    
    This endpoint mimics the behavior of OpenAI's /v1/chat/completions endpoint.
    """
    try:
        # Check if model exists
        model_path = settings.OUTPUT_DIR / request.model
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {request.model}"
            )
        
        # Generate a unique ID for this completion
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created_timestamp = int(time.time())
        
        # If streaming is requested, use streaming response
        if request.stream:
            return StreamingResponse(
                generate_chat_completion(
                    model_path=model_path,
                    messages=request.messages,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    max_tokens=request.max_tokens,
                    stop=request.stop,
                    stream=True,
                    completion_id=completion_id,
                    created=created_timestamp
                ),
                media_type="text/event-stream"
            )
        
        # Otherwise, generate text and return as JSON
        result = await generate_chat_completion(
            model_path=model_path,
            messages=request.messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop,
            stream=False,
            completion_id=completion_id,
            created=created_timestamp
        )
        
        return result
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat completion failed: {str(e)}"
        )

@router.get("/models", response_model=Dict[str, Any])
async def list_models():
    """List all available models in OpenAI format."""
    try:
        models_data = []
        for model_dir in settings.OUTPUT_DIR.glob("*"):
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                # Get model metadata if available
                metadata_file = model_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                
                # Format in OpenAI style
                models_data.append({
                    "id": model_dir.name,
                    "object": "model",
                    "created": int(time.mktime(time.strptime(metadata.get("created_at", "2023-01-01T00:00:00"), "%Y-%m-%dT%H:%M:%S.%f"))) if "created_at" in metadata else int(time.time()),
                    "owned_by": "vana",
                    "permission": [],
                    "root": metadata.get("base_model", "unknown"),
                    "parent": None
                })
        
        return {
            "object": "list",
            "data": models_data
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )