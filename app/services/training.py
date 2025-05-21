"""
Training Service Module - Model fine-tuning and training job management.

This module provides functionality for starting training jobs, monitoring
their status, and retrieving training events and results.

Examples:
    Starting a training job:
    
    >>> from app.services.training import TrainingService
    >>> from app.models.training import TrainingRequest, TrainingParameters
    >>> from fastapi import BackgroundTasks
    >>>
    >>> training_service = TrainingService()
    >>> background_tasks = BackgroundTasks()
    >>>
    >>> # Create request with existing query_id
    >>> request = TrainingRequest(
    ...     model_name="meta-llama/Llama-2-7b-hf",
    ...     query_id="query_12345",
    ...     training_params=TrainingParameters(num_epochs=3, batch_size=8)
    ... )
    >>>
    >>> # Start training
    >>> response = training_service.start_training_job(request, background_tasks)
    >>> print(f"Job started: {response.job_id}")
    
    Monitoring job status:
    
    >>> status = training_service.get_training_job_status("train_abcd1234")
    >>> print(f"Training status: {status.status}, Progress: {status.progress}%")
    
    Retrieving training events:
    
    >>> events = await training_service.get_training_events("train_abcd1234")
    >>> for event in events:
    ...     if event.type == "progress":
    ...         print(f"Progress: {event.data.get('progress')}%")
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import HTTPException, status, BackgroundTasks, Request

from app.models.training import TrainingStatus, TrainingRequest, TrainingResponse, TrainingParameters, TrainingEvent
from app.config import settings as global_settings
from app.utils.query_engine_client import QueryEngineClient
from app.ml.trainer import train_model
from app.utils.events import format_sse_event, subscribe_to_training_events, get_training_events

class TrainingService:
    """Service for model training operations."""
    
    def __init__(self, settings=None, query_client=None):
        """
        Initialize the training service.
        
        Args:
            settings: Application settings
            query_client: Query engine client for data access
        """        
        self.settings = settings or global_settings
        self.query_client = query_client or QueryEngineClient()
    
    def start_training_job(
        self, 
        request: TrainingRequest,
        background_tasks: BackgroundTasks
    ) -> TrainingResponse:
        """
        Start a new training job.
        
        Args:
            request: Training request parameters
            background_tasks: FastAPI background tasks
            
        Returns:
            Training response with job details
            
        Raises:
            HTTPException: If validation fails
        """
        # Validate the request
        self.validate_training_request(request)
        
        # Generate job ID and setup
        job_id = self.generate_job_id()
        output_path = self.get_output_path(request, job_id)
        training_params = self.get_training_parameters(request.training_params)
        
        # Handle existing query ID or run a new query
        query_id = request.query_id
        if not query_id:
            query_id = self.execute_new_query(request)
        
        # Start the actual training in background
        background_tasks.add_task(
            train_model,
            job_id=job_id,
            query_id=query_id,
            model_name=request.model_name,
            output_dir=output_path,
            training_params=training_params
        )
        
        # Return the response
        return TrainingResponse(
            job_id=job_id,
            query_id=query_id,
            status="started",
            message=f"Training job started. Connect to /train/{job_id}/events for real-time updates."
        )
    
    def get_training_job_status(self, job_id: str) -> TrainingStatus:
        """
        Get the status of a training job.
        
        Args:
            job_id: The training job ID
            
        Returns:
            TrainingStatus with current job status
            
        Raises:
            HTTPException: If job not found
        """
        return self.load_training_status(job_id)
    
    async def get_training_events(self, job_id: str) -> List[TrainingEvent]:
        """
        Get all events for a training job.
        
        Args:
            job_id: The training job ID
            
        Returns:
            List of training events
            
        Raises:
            HTTPException: If job not found
        """
        self._check_job_exists(job_id)
        
        events = await get_training_events(job_id)
        return [TrainingEvent(**event) for event in events]
    
    def _check_job_exists(self, job_id: str) -> None:
        """
        Check if a job exists and raise exception if not.
        
        Args:
            job_id: The training job ID
            
        Raises:
            HTTPException: If job not found
        """
        status_file = self.settings.WORKING_DIR / f"{job_id}_status.json"
        if not status_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found"
            )
    
    def load_training_status(self, training_job_id: str) -> TrainingStatus:
        """
        Load the status of a training job from its status file.
        
        Args:
            training_job_id: ID of the training job
            
        Returns:
            Training status data
            
        Raises:
            HTTPException: If status file doesn't exist or can't be loaded
        """
        status_file = self.settings.WORKING_DIR / f"{training_job_id}_status.json"
        
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
    
    def validate_training_request(self, request: TrainingRequest) -> None:
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

    def generate_job_id(self) -> str:
        """
        Generate a unique job ID for training.
        
        Returns:
            A unique training job ID
        """
        return f"train_{os.urandom(4).hex()}"

    def get_output_path(self, request: TrainingRequest, job_id: str) -> Path:
        """
        Get the full output path for the trained model.
        
        Args:
            request: The training request
            job_id: The unique job ID
            
        Returns:
            Path to the output directory
        """
        output_dir = request.output_dir or f"model_{job_id}"
        return self.settings.OUTPUT_DIR / output_dir

    def get_training_parameters(self, request_params: Optional[TrainingParameters]) -> Dict[str, Any]:
        """
        Merge default and custom training parameters.
        
        Args:
            request_params: Custom training parameters from the request
            
        Returns:
            Dictionary of training parameters
        """
        # Start with default parameters
        params = {
            "num_epochs": self.settings.NUM_EPOCHS,
            "learning_rate": self.settings.LEARNING_RATE,
            "batch_size": self.settings.BATCH_SIZE,
            "max_seq_length": self.settings.MAX_SEQ_LENGTH
        }
        
        # Update with custom parameters if provided
        if request_params:
            params.update(request_params.dict())
        
        return params

    async def execute_new_query(self, request: TrainingRequest) -> str:
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
        
        result = self.query_client.execute_query(
            job_id=request.query_params.compute_job_id,
            refiner_id=request.query_params.refiner_id,
            query=request.query_params.query,
            query_signature=request.query_params.query_signature,
            results_dir=self.settings.WORKING_DIR,
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