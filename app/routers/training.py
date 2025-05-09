from fastapi import APIRouter, BackgroundTasks, HTTPException, Depends, status, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from pathlib import Path
import json
import logging

from config import settings
from models.trainer import train_model
from utils.db import get_training_data
from utils.events import subscribe_to_training_events, format_sse_event, get_training_events
from utils.query_engine_client import QueryEngineClient

logger = logging.getLogger(__name__)
router = APIRouter()

class QueryParams(BaseModel):
    """Parameters for a query to the database."""
    query: str
    params: Optional[List[Any]] = None
    refiner_id: Optional[int] = None
    query_signature: Optional[str] = None

class TrainingRequest(BaseModel):
    """Training request model."""
    model_name: Optional[str] = settings.DEFAULT_BASE_MODEL
    output_dir: Optional[str] = None
    training_params: Optional[Dict[str, Any]] = None
    query_id: Optional[str] = None  # ID of an existing query
    query_params: Optional[QueryParams] = None  # Parameters for a new query
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "output_dir": "my_finetuned_model",
                "training_params": {
                    "num_epochs": 3,
                    "learning_rate": 2e-4,
                    "batch_size": 4
                },
                "query_params": {
                    "query": "SELECT * FROM tweets WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
                    "params": ["user123"],
                    "refiner_id": 12
                }
            }
        }

class TrainingResponse(BaseModel):
    """Training response model."""
    job_id: str
    query_id: str
    status: str
    message: str

@router.post("/", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def train(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a training job to fine-tune a model using query results.
    
    This endpoint will:
    1. Use the provided query_id or execute a new query with query_params
    2. Extract training data from the query results in the input database
    3. Fine-tune the specified model using Unsloth
    4. Save the fine-tuned model to the output directory
    
    Either query_id or query_params must be provided to identify the data to use for training.
    
    Training progress can be monitored in real-time by connecting to the /train/{job_id}/events endpoint.
    """
    try:
        # Validate that either query_id or query_params is provided
        if not request.query_id and not request.query_params:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either query_id or query_params must be provided"
            )
        
        # Generate a unique job ID
        job_id = f"train_{os.urandom(4).hex()}"
        
        # Set output directory
        output_dir = request.output_dir or f"model_{job_id}"
        full_output_path = settings.OUTPUT_DIR / output_dir
        
        # Merge default and custom training parameters
        training_params = {
            "num_epochs": settings.NUM_EPOCHS,
            "learning_rate": settings.LEARNING_RATE,
            "batch_size": settings.BATCH_SIZE,
            "max_seq_length": settings.MAX_SEQ_LENGTH
        }
        if request.training_params:
            training_params.update(request.training_params)
        
        if request.query_id:
            logger.info(f"Training with existing query_id: {request.query_id}")

            # Start training in background
            background_tasks.add_task(
                train_model,
                job_id=job_id,
                query_id=request.query_id,
                model_name=request.model_name,
                output_dir=full_output_path,
                training_params=training_params
            )
            
            return TrainingResponse(
                job_id=job_id,
                query_id=request.query_id,
                status="started",
                message=f"Training job started. Connect to /train/{job_id}/events for real-time updates."
            )

        client = QueryEngineClient()
        result = client.execute_query(
            job_id=job_id,
            refiner_id=request.query_params.refiner_id,
            query=request.query_params.query,
            query_signature=request.query_params.query_signature,
            results_dir=settings.WORKING_DIR
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
        
        # Start training in background
        background_tasks.add_task(
            train_model,
            job_id=job_id,
            query_id=query_id,
            model_name=request.model_name,
            output_dir=full_output_path,
            training_params=training_params
        )
        
        return TrainingResponse(
            job_id=job_id,
            query_id=query_id,
            status="started",
            message=f"Training job started. Connect to /train/{job_id}/events for real-time updates."
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start training job: {str(e)}"
        )

@router.get("/{job_id}", response_model=Dict[str, Any])
async def get_training_status(job_id: str):
    """Get the status of a training job."""
    status_file = settings.WORKING_DIR / f"{job_id}_status.json"
    
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    try:
        with open(status_file, "r") as f:
            status_data = json.load(f)
        return status_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get training status: {str(e)}"
        )

@router.get("/{job_id}/events")
async def stream_training_events(job_id: str, request: Request):
    """
    Stream training events for a job using Server-Sent Events (SSE).
    
    This endpoint allows clients to receive real-time updates about the training process.
    The connection remains open until the training is complete or an error occurs.
    
    Events include:
    - progress: Updates on training progress
    - log: Log messages from the training process
    - complete: Sent when training is complete
    - error: Sent if an error occurs during training
    """
    # Check if the job exists
    status_file = settings.WORKING_DIR / f"{job_id}_status.json"
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    async def event_generator():
        try:
            # Send headers for SSE
            yield "retry: 1000\n\n"
            
            # Stream events
            async for event in subscribe_to_training_events(job_id):
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
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )

@router.get("/{job_id}/events/history", response_model=List[Dict[str, Any]])
async def get_training_event_history(job_id: str):
    """
    Get the history of training events for a job.
    
    This endpoint returns all events that have been emitted for the training job,
    allowing clients to catch up on progress if they weren't connected from the start.
    """
    # Check if the job exists
    status_file = settings.WORKING_DIR / f"{job_id}_status.json"
    if not status_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Training job {job_id} not found"
        )
    
    events = await get_training_events(job_id)
    return events