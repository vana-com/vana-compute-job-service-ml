from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import json

from app.config import settings
from app.models.inference import generate_text, load_model

router = APIRouter()

class InferenceRequest(BaseModel):
    """Inference request model."""
    model_path: str
    prompt: str
    max_new_tokens: Optional[int] = settings.MAX_NEW_TOKENS
    temperature: Optional[float] = settings.TEMPERATURE
    top_p: Optional[float] = settings.TOP_P
    stream: Optional[bool] = False
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_path": "my_finetuned_model",
                "prompt": "What is machine learning?",
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "stream": False
            }
        }

class InferenceResponse(BaseModel):
    """Inference response model."""
    text: str
    model_path: str
    generation_time: float

@router.post("/", response_model=InferenceResponse)
async def inference(request: InferenceRequest):
    """
    Generate text using a fine-tuned model.
    
    This endpoint will:
    1. Load the specified model
    2. Generate text based on the provided prompt
    3. Return the generated text
    """
    try:
        # Check if model exists
        model_path = settings.OUTPUT_DIR / request.model_path
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found at {request.model_path}"
            )
        
        # If streaming is requested, use streaming response
        if request.stream:
            return StreamingResponse(
                generate_text(
                    model_path=model_path,
                    prompt=request.prompt,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    stream=True
                ),
                media_type="text/event-stream"
            )
        
        # Otherwise, generate text and return as JSON
        result = generate_text(
            model_path=model_path,
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            stream=False
        )
        
        return InferenceResponse(
            text=result["text"],
            model_path=str(model_path),
            generation_time=result["generation_time"]
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )

@router.get("/models", response_model=List[Dict[str, Any]])
async def list_models():
    """List all available fine-tuned models."""
    try:
        models = []
        for model_dir in settings.OUTPUT_DIR.glob("*"):
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                # Get model metadata if available
                metadata_file = model_dir / "metadata.json"
                metadata = {}
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                
                models.append({
                    "name": model_dir.name,
                    "path": str(model_dir),
                    "created_at": metadata.get("created_at", "unknown"),
                    "base_model": metadata.get("base_model", "unknown"),
                    "metadata": metadata
                })
        
        return models
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )