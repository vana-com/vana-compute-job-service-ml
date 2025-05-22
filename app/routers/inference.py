from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
import logging

from app.ml.inference import generate_chat_completion
from app.models.openai import ChatCompletionRequest, ChatCompletionResponse
from app.models.inference import ModelListResponse, ModelData
from app.services import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter()
inference_service = InferenceService()

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
        model_path = inference_service.get_model_path(request.model)
        
        # Generate IDs and timestamps
        completion_id = inference_service.generate_completion_id()
        created_timestamp = inference_service.get_current_timestamp()
        
        # If streaming is requested, use streaming response
        if request.stream:
            # For streaming, return a streaming response
            logger.info(f"Streaming chat completion for model {request.model}")
            return StreamingResponse(
                await generate_chat_completion(
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
    except FileNotFoundError as e:
        # Model not found
        error_msg = f"Model '{request.model}' not found: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=error_msg
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        error_msg = f"Error generating chat completion with model '{request.model}': {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
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
        logger.info("Listing available models")
        return inference_service.list_available_models()
    except Exception as e:
        error_msg = f"Error listing models: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_msg
        )