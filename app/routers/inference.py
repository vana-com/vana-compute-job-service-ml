from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List
import time
import json
import uuid
from pathlib import Path

from app.config import settings
from app.ml.inference import generate_chat_completion
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse
from app.models.inference import ModelListResponse, ModelData

router = APIRouter()

def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model directory.
    
    Args:
        model_name: Name of the model to find
        
    Returns:
        Path to the model directory
        
    Raises:
        HTTPException: If the model is not found
    """
    model_path = settings.OUTPUT_DIR / model_name
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}"
        )
    return model_path

def generate_completion_id() -> str:
    """
    Generate a unique ID for a completion.
    
    Returns:
        Unique completion ID string
    """
    return f"chatcmpl-{uuid.uuid4().hex}"

def get_current_timestamp() -> int:
    """
    Get the current Unix timestamp.
    
    Returns:
        Current time as Unix timestamp (integer)
    """
    return int(time.time())

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Create a chat completion following the OpenAI API specification.
    
    Args:
        request: The chat completion request containing messages and parameters
    
    Returns:
        Either a streaming response or a JSON response with the completion
        
    Raises:
        HTTPException: If model is not found or completion fails
    """
    try:
        # Check if model exists and get path
        model_path = get_model_path(request.model)
        
        # Generate IDs and timestamps
        completion_id = generate_completion_id()
        created_timestamp = get_current_timestamp()
        
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

def load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    """
    Load metadata for a model from its metadata.json file.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Dictionary containing model metadata, or empty dict if not found
    """
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
        
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def parse_model_timestamp(timestamp_str: str) -> int:
    """
    Parse a timestamp string into a Unix timestamp.
    
    Args:
        timestamp_str: Timestamp string in ISO format
        
    Returns:
        Unix timestamp as integer
    """
    try:
        return int(time.mktime(time.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")))
    except Exception:
        return get_current_timestamp()

def create_model_data(model_dir: Path) -> ModelData:
    """
    Create a ModelData object for a model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        ModelData object with model information
    """
    metadata = load_model_metadata(model_dir)
    
    # Get created timestamp
    created = get_current_timestamp()
    if "created_at" in metadata:
        created = parse_model_timestamp(metadata["created_at"])
    
    # Get root model name
    root = metadata.get("base_model", "unknown")
    
    return ModelData(
        id=model_dir.name,
        created=created,
        root=root
    )

@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    List all available models in OpenAI format.
    
    Returns:
        ModelListResponse containing a list of available models
        
    Raises:
        HTTPException: If there's an error listing models
    """
    try:
        models_data: List[ModelData] = []
        
        for model_dir in settings.OUTPUT_DIR.glob("*"):
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models_data.append(create_model_data(model_dir))
        
        return ModelListResponse(data=models_data)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )