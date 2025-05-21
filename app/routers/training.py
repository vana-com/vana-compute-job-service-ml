from fastapi import APIRouter, BackgroundTasks, HTTPException, status, Request
from fastapi.responses import StreamingResponse
from typing import List
import logging

from app.config import settings
from app.models.training import (
    TrainingRequest, TrainingResponse, TrainingStatus, 
    TrainingEvent
)
from app.utils.events import get_training_events
from app.services import TrainingService

logger = logging.getLogger(__name__)
router = APIRouter()
training_service = TrainingService()

@router.post("", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED, name="train")
async def train(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Start a training job to fine-tune a model using query results.
    
    This endpoint will:
    1. Use the provided query_id or execute a new query with query_params
    2. Extract training data from the query results in the input database
    3. Fine-tune the specified model using Unsloth
    4. Save the fine-tuned model to the output directory
    
    Either query_id or query_params must be provided to identify the data to use for training.
    
    Training progress can be monitored in real-time by connecting to the /train/{training_job_id}/events endpoint.
    
    Args:
        request: Training request with model and data specifications
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        Training response with job information
        
    Raises:
        HTTPException: If validation fails or training can't be started
    """
    return training_service.start_training_job(request, background_tasks)


@router.get("/{training_job_id}", response_model=TrainingStatus)
async def get_training_status(training_job_id: str) -> TrainingStatus:
    """
    Get the status of a training job.
    
    Args:
        training_job_id: ID of the training job
        
    Returns:
        Status of the training job
        
    Raises:
        HTTPException: If job not found or status loading fails
    """
    return training_service.get_training_job_status(training_job_id)

@router.get("/{training_job_id}/events")
async def stream_training_events(training_job_id: str, request: Request) -> StreamingResponse:
    """
    Stream training events for a job using Server-Sent Events (SSE).
    
    This endpoint allows clients to receive real-time updates about the training process.
    The connection remains open until the training is complete or an error occurs.
    
    Events include:
    - progress: Updates on training progress
    - log: Log messages from the training process
    - complete: Sent when training is complete
    - error: Sent if an error occurs during training
    
    Args:
        training_job_id: ID of the training job
        request: FastAPI request object
        
    Returns:
        Streaming response with SSE events
        
    Raises:
        HTTPException: If job not found
    """
    # Check if the job exists
    status_file = settings.WORKING_DIR / f"{training_job_id}_status.json"
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {training_job_id} not found"
        )
    
    return StreamingResponse(
        training_service.generate_sse_events(training_job_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )

@router.get("/{training_job_id}/events/history", response_model=List[TrainingEvent])
async def get_training_event_history(training_job_id: str) -> List[TrainingEvent]:
    """
    Get the history of training events for a job.
    
    This endpoint returns all events that have been emitted for the training job,
    allowing clients to catch up on progress if they weren't connected from the start.
    
    Args:
        training_job_id: ID of the training job
        
    Returns:
        List of training events
        
    Raises:
        HTTPException: If job not found
    """
    # Check if the job exists
    status_file = settings.WORKING_DIR / f"{training_job_id}_status.json"
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {training_job_id} not found"
        )
    
    events = await get_training_events(training_job_id)
    return [TrainingEvent(**event) for event in events]