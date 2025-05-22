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
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List

from fastapi import HTTPException, status, BackgroundTasks, Request

from app.models.training import TrainingStatus, TrainingRequest, TrainingResponse, TrainingParameters, TrainingEvent
from app.config import settings as global_settings
from app.utils.query_engine_client import QueryEngineClient
from app.ml.trainer import train_model
from app.utils.events import format_sse_event, subscribe_to_training_events, get_training_events

# Set up logging
logger = logging.getLogger(__name__)

class TrainingService:
    """Service for model training operations."""
    
    def __init__(self, settings=None, query_client=None):
        """
        Initialize the training service.
        
        Args:
            settings: Application settings
            query_client: Query engine client for data access
        """        
        logger.info("Initializing TrainingService")
        self.settings = settings or global_settings
        self.query_client = query_client or QueryEngineClient()
    
    async def start_training_job(
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
        logger.info(f"Starting training job for model: {request.model_name}")
        
        try:
            # Validate the request
            logger.debug("Validating training request")
            self.validate_training_request(request)
            
            # Generate job ID and setup
            job_id = self.generate_job_id()
            logger.info(f"Generated job ID: {job_id}")
            
            working_path = self.get_working_path(request, job_id)
            logger.info(f"Model will be trained and saved to: {working_path}")

            output_path = self.get_output_path(request, job_id)
            logger.info(f"Model artifact will be saved to: {output_path}")
            
            training_params = self.get_training_parameters(request.training_params)
            logger.debug(f"Training parameters: {training_params}")
            
            # Handle existing query ID or run a new query
            query_id = request.query_id
            if query_id:
                logger.info(f"Using existing query ID: {query_id}")
            else:
                logger.info("No query ID provided, executing new query")
                query_id = await self.execute_new_query(request)
                logger.info(f"Query executed successfully, query ID: {query_id}")
            
            # Start the actual training in background
            logger.info(f"Starting background training task for job {job_id}")
            background_tasks.add_task(
                train_model,
                job_id=job_id,
                query_id=query_id,
                model_name=request.model_name,
                working_path=working_path,
                output_path=output_path,
                training_params=training_params
            )
            
            # Return the response
            response = TrainingResponse(
                job_id=job_id,
                query_id=query_id,
                status="started",
                message=f"Training job started. Connect to /train/{job_id}/events for real-time updates."
            )
            logger.info(f"Training job {job_id} successfully started")
            return response
            
        except HTTPException:
            # Re-raise HTTP exceptions directly
            logger.error(f"HTTP exception in start_training_job: {traceback.format_exc()}")
            raise
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error starting training job: {str(e)}\n{error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start training job: {str(e)}"
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
        logger.info(f"Getting status for training job: {job_id}")
        try:
            status = self.load_training_status(job_id)
            logger.debug(f"Job {job_id} status: {status.status}")
            return status
        except HTTPException:
            # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error getting training status for job {job_id}: {str(e)}\n{error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get training status: {str(e)}"
            )
    
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
        logger.info(f"Getting events for training job: {job_id}")
        try:
            self._check_job_exists(job_id)
            
            events = await get_training_events(job_id)
            logger.debug(f"Retrieved {len(events)} events for job {job_id}")
            
            return [TrainingEvent(**event) for event in events]
        except HTTPException:
            # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error getting training events for job {job_id}: {str(e)}\n{error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to get training events: {str(e)}"
            )
    
    def _check_job_exists(self, job_id: str) -> None:
        """
        Check if a job exists and raise exception if not.
        
        Args:
            job_id: The training job ID
            
        Raises:
            HTTPException: If job not found
        """
        logger.debug(f"Checking if job exists: {job_id}")
        status_file = self.settings.WORKING_DIR / f"{job_id}_status.json"
        
        if not status_file.exists():
            logger.error(f"Training job {job_id} not found - status file missing at {status_file}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {job_id} not found. Check the job ID and ensure the job was created successfully."
            )
        
        logger.debug(f"Job {job_id} exists with status file at {status_file}")
    
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
        logger.debug(f"Loading status for job: {training_job_id}")
        status_file = self.settings.WORKING_DIR / f"{training_job_id}_status.json"
        
        if not status_file.exists():
            logger.error(f"Training job {training_job_id} status file not found at: {status_file}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Training job {training_job_id} not found. Check the job ID and ensure the job was created successfully."
            )
        
        try:
            with open(status_file, "r") as f:
                status_data = json.load(f)
                
            logger.debug(f"Loaded status for job {training_job_id}: {status_data.get('status', 'unknown')}")
            return TrainingStatus(**status_data)
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in status file for job {training_job_id}: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to parse job status file: {str(e)}"
            )
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error loading status for job {training_job_id}: {str(e)}\n{error_detail}")
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
        logger.debug("Validating training request")
        validation_errors = []
        
        if not request.query_id and not request.query_params:
            msg = "Either query_id or query_params must be provided"
            logger.error(f"Validation error: {msg}")
            validation_errors.append(msg)
            
        if request.query_params and not request.query_params.query:
            msg = "Query string is required when providing query_params"
            logger.error(f"Validation error: {msg}")
            validation_errors.append(msg)
            
        if validation_errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=", ".join(validation_errors)
            )
            
        logger.debug("Training request validation successful")

    def generate_job_id(self) -> str:
        """
        Generate a unique job ID for training.
        
        Returns:
            A unique training job ID
        """
        job_id = f"train_{os.urandom(4).hex()}"
        logger.debug(f"Generated job ID: {job_id}")
        return job_id

    def get_output_path(self, request: TrainingRequest, job_id: str) -> Path:
        """
        Get the path where the zipped model artifact will be saved.
        This is for artifact scanning and Compute Engine API download.
        
        Args:
            request: The training request
            job_id: The unique job ID
            
        Returns:
            Path to the output zip file
        """
        model_name = request.output_dir or f"model_{job_id}"
        # Add .zip extension to the model output file
        zip_name = f"{model_name}.zip"
        output_path = self.settings.OUTPUT_DIR / zip_name
        
        # Make sure the output directory exists
        if not self.settings.OUTPUT_DIR.exists():
            logger.info(f"Creating output directory: {self.settings.OUTPUT_DIR}")
            self.settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            
        logger.debug(f"Output zip path for job {job_id}: {output_path}")
        return output_path

    def get_working_path(self, request: TrainingRequest, job_id: str) -> Path:
        """
        Get the path where the model will be trained and saved during training.
        This is the working version used for inference.
        
        Args:
            request: The training request
            job_id: The unique job ID
            
        Returns:
            Path to the working directory for model files
        """
        model_name = request.output_dir or f"model_{job_id}"
        working_path = self.settings.WORKING_DIR / model_name
        
        # Make sure the working directory exists
        if not self.settings.WORKING_DIR.exists():
            logger.info(f"Creating working directory: {self.settings.WORKING_DIR}")
            self.settings.WORKING_DIR.mkdir(parents=True, exist_ok=True)
            
        logger.debug(f"Working path for job {job_id}: {working_path}")
        return working_path

    
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
        
        if request_params:
            logger.debug(f"Using custom training parameters: {request_params.dict()}")
            custom_params = request_params.dict(exclude_unset=True)
            params.update(custom_params)
        else:
            logger.debug(f"Using default training parameters: {params}")
        
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
            logger.error("Query parameters are required but not provided")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query parameters are required but not provided"
            )
        
        logger.info(f"Executing query: {request.query_params.query}")
        logger.debug(f"Query parameters: {request.query_params.dict()}")
        
        try:
            result = self.query_client.execute_query(
                job_id=request.query_params.compute_job_id,
                refiner_id=request.query_params.refiner_id,
                query=request.query_params.query,
                query_signature=request.query_params.query_signature,
                results_dir=self.settings.WORKING_DIR,
                params=request.query_params.params
            )

            if not result.success:
                logger.error(f"Query execution failed: {result.error}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Failed to execute query: {result.error}"
                )
            
            query_id = result.data.get("query_id")
            if not query_id:
                logger.error("No query ID returned from query execution")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="No query ID returned from query execution"
                )
            
            logger.info(f"Query executed successfully, query ID: {query_id}")
            return query_id
            
        except HTTPException:
            # Re-raise HTTP exceptions directly
            raise
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error executing query: {str(e)}\n{error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to execute query: {str(e)}"
            )

# Helper function - not part of the service class
async def generate_sse_events(training_job_id: str, request: Request):
    """
    Generate Server-Sent Events (SSE) for a training job.
    
    Args:
        training_job_id: ID of the training job
        request: FastAPI request object
        
    Yields:
        SSE formatted events
    """
    logger.info(f"Starting SSE event stream for job: {training_job_id}")
    try:
        # Send headers for SSE
        yield "retry: 1000\n\n"
        event_count = 0
        
        # Stream events
        async for event in subscribe_to_training_events(training_job_id):
            event_count += 1
            formatted_event = format_sse_event(event)
            logger.debug(f"Sending event {event_count} of type {event['type']} for job {training_job_id}")
            
            yield formatted_event
            
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info(f"Client disconnected from SSE stream for job {training_job_id}")
                break
                
            # If this is a completion or error event, log it
            if event["type"] in ["complete", "error"]:
                logger.info(f"Training job {training_job_id} {event['type']} event sent. Streaming will end.")
                
        logger.info(f"SSE stream for job {training_job_id} ended after {event_count} events")
    except Exception as e:
        error_detail = traceback.format_exc()
        logger.error(f"Error in SSE stream for job {training_job_id}: {str(e)}\n{error_detail}")
        
        # Send error event
        error_event = {
            "type": "error",
            "data": {"message": f"Error streaming events: {str(e)}"}
        }
        yield format_sse_event(error_event)