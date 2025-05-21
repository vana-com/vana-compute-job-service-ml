import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import HTTPException, status, BackgroundTasks, Request

from app.models.training import TrainingStatus, TrainingRequest, TrainingResponse, TrainingParameters
from app.config import settings
from app.utils.query_engine_client import QueryEngineClient
from app.ml.trainer import train_model
from app.utils.events import format_sse_event, subscribe_to_training_events

def load_training_status(training_job_id: str) -> TrainingStatus:
    """
    Load the status of a training job from its status file.
    
    Args:
        training_job_id: ID of the training job
        
    Returns:
        Training status data
        
    Raises:
        HTTPException: If status file doesn't exist or can't be loaded
    """
    status_file = settings.WORKING_DIR / f"{training_job_id}_status.json"
    
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {training_job_id} not found"
        )
    
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)
        return TrainingStatus(**status_data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )
    
def validate_training_request(request: TrainingRequest) -> None:
    """
    Validate that the training request contains required data.
    
    Args:
        request: The training request to validate
        
    Raises:
        HTTPException: If validation fails
    """
    if not request.query_id and not request.query_params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either query_id or query_params must be provided"
        )

def generate_job_id() -> str:
    """
    Generate a unique job ID for training.
    
    Returns:
        A unique training job ID
    """
    return f"train_{os.urandom(4).hex()}"

def get_output_path(request: TrainingRequest, job_id: str) -> Path:
    """
    Get the full output path for the trained model.
    
    Args:
        request: The training request
        job_id: The unique job ID
        
    Returns:
        Path to the output directory
    """
    output_dir = request.output_dir or f"model_{job_id}"
    return settings.OUTPUT_DIR / output_dir

def get_training_parameters(request_params: Optional[TrainingParameters]) -> Dict[str, Any]:
    """
    Merge default and custom training parameters.
    
    Args:
        request_params: Custom training parameters from the request
        
    Returns:
        Dictionary of training parameters
    """
    # Start with default parameters
    params = {
        "num_epochs": settings.NUM_EPOCHS,
        "learning_rate": settings.LEARNING_RATE,
        "batch_size": settings.BATCH_SIZE,
        "max_seq_length": settings.MAX_SEQ_LENGTH
    }
    
    # Update with custom parameters if provided
    if request_params:
        params.update(request_params.dict())
    
    return params

async def execute_new_query(request: TrainingRequest) -> str:
    """
    Execute a new query to get training data.
    
    Args:
        request: The training request containing query parameters
        
    Returns:
        The query ID
        
    Raises:
        HTTPException: If query execution fails
    """
    if not request.query_params:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query parameters are required but not provided"
        )
    
    client = QueryEngineClient()
    result = client.execute_query(
        job_id=request.query_params.compute_job_id,
        refiner_id=request.query_params.refiner_id,
        query=request.query_params.query,
        query_signature=request.query_params.query_signature,
        results_dir=settings.WORKING_DIR,
        params=request.query_params.params
    )

    if not result.success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to execute query: {result.error}"
        )
    
    query_id = result.data.get("query_id")
    if not query_id:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No query ID returned from query execution"
        )
    
    return query_id

def start_training_job(
    background_tasks: BackgroundTasks,
    job_id: str,
    query_id: str,
    model_name: str,
    output_path: Path,
    training_params: Dict[str, Any]
) -> TrainingResponse:
    """
    Start a training job in the background.
    
    Args:
        background_tasks: FastAPI background tasks
        job_id: Unique job ID
        query_id: ID of the query with training data
        model_name: Model to fine-tune
        output_path: Path to save the model
        training_params: Training parameters
        
    Returns:
        Training response with job information
    """
    # Start training in background
    background_tasks.add_task(
        train_model,
        job_id=job_id,
        query_id=query_id,
        model_name=model_name,
        output_dir=output_path,
        training_params=training_params
    )
    
    # Return response
    return TrainingResponse(
        job_id=job_id,
        query_id=query_id,
        status="started",
        message=f"Training job started. Connect to /train/{job_id}/events for real-time updates."
    )

async def generate_sse_events(training_job_id: str, request: Request):
    """
    Generate Server-Sent Events (SSE) for a training job.
    
    Args:
        training_job_id: ID of the training job
        request: FastAPI request object
        
    Yields:
        SSE formatted events
    """
    try:
        # Send headers for SSE
        yield "retry: 1000\n\n"
        
        # Stream events
        async for event in subscribe_to_training_events(training_job_id):
            yield format_sse_event(event)
            
            # Check if client disconnected
            if await request.is_disconnected():
                break
    except Exception as e:
        # Send error event
        error_event = {
            "type": "error",
            "data": {"message": f"Error streaming events: {str(e)}"}
        }
        yield format_sse_event(error_event)