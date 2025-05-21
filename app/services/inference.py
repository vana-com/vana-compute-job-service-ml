"""
Inference Service Module - Model management and inference capabilities.

This module provides functionality for accessing models, generating completions,
and managing model metadata.

Examples:
    Basic model listing:
    
    >>> from app.services.inference import InferenceService
    >>> inference_service = InferenceService()
    >>> models = inference_service.list_available_models()
    >>> for model in models.data:
    ...     print(f"Model: {model.id}, Base: {model.root}")
    
    Working with models:
    
    >>> model_path = inference_service.get_model_path("my_finetuned_model")
    >>> metadata = inference_service.load_model_metadata(model_path)
    >>> print(f"Model trained on: {metadata.get('created_at')}")
    
    Generation utilities:
    
    >>> completion_id = inference_service.generate_completion_id()
    >>> timestamp = inference_service.get_current_timestamp()
"""

import uuid
import time
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import HTTPException, status
from app.models.inference import ModelData, ModelListResponse

# Set up logging
logger = logging.getLogger(__name__)

class InferenceService:
    """Service for model inference and model management."""
    
    def __init__(self, settings=None):
        """
        Initialize the inference service.
        
        Args:
            settings: Application settings, defaults to global settings
        """
        logger.info("Initializing InferenceService")
        from app.config import settings as global_settings
        self.settings = settings or global_settings
    
    def get_model_path(self, model_name: str) -> Path:
        """
        Get path to a model directory.
        
        Args:
            model_name: Name of the model to find
            
        Returns:
            Path to the model directory
            
        Raises:
            HTTPException: If the model is not found
        """
        logger.debug(f"Looking for model: {model_name}")
        model_path = self.settings.OUTPUT_DIR / model_name
        
        if not model_path.exists():
            logger.error(f"Model not found: {model_name} (path: {model_path})")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_name}. Check the model name and ensure it has been properly trained or downloaded."
            )
            
        logger.debug(f"Found model at: {model_path}")
        return model_path
    
    def generate_completion_id(self) -> str:
        """
        Generate a unique ID for a completion.
        
        Returns:
            Unique completion ID string
        """
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        logger.debug(f"Generated completion ID: {completion_id}")
        return completion_id
    
    def get_current_timestamp(self) -> int:
        """
        Get current Unix timestamp.
        
        Returns:
            Current time as Unix timestamp (integer)
        """
        timestamp = int(time.time())
        return timestamp
    
    def list_available_models(self) -> ModelListResponse:
        """
        List all available models.
        
        Returns:
            ModelListResponse containing available models
        """
        logger.info(f"Listing available models in {self.settings.OUTPUT_DIR}")
        models_data: List[ModelData] = []
        
        try:
            # Ensure the output directory exists
            if not self.settings.OUTPUT_DIR.exists():
                logger.warning(f"Output directory doesn't exist: {self.settings.OUTPUT_DIR}")
                self.settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created output directory: {self.settings.OUTPUT_DIR}")
                
            # Count how many directories we check
            checked_dirs = 0
            valid_models = 0
            
            for model_dir in self.settings.OUTPUT_DIR.glob("*"):
                checked_dirs += 1
                if model_dir.is_dir() and (model_dir / "config.json").exists():
                    logger.debug(f"Found valid model at: {model_dir}")
                    valid_models += 1
                    models_data.append(self.create_model_data(model_dir))
                else:
                    logger.debug(f"Skipping invalid model directory: {model_dir}")
            
            logger.info(f"Found {valid_models} valid models out of {checked_dirs} directories")
            
            return ModelListResponse(data=models_data)
            
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error listing models: {str(e)}\n{error_detail}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to list models: {str(e)}"
            )
    
    def load_model_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """
        Load model metadata from file.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            Dictionary containing model metadata, or empty dict if not found
        """
        metadata_file = model_dir / "metadata.json"
        logger.debug(f"Loading metadata from: {metadata_file}")
        
        if not metadata_file.exists():
            logger.info(f"No metadata file found at: {metadata_file}")
            return {}
        
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                logger.debug(f"Loaded metadata: {list(metadata.keys())}")
                return metadata
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in metadata file {metadata_file}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Error reading metadata file {metadata_file}: {str(e)}")
            return {}
    
    def parse_model_timestamp(self, timestamp_str: str) -> int:
        """
        Parse timestamp string to Unix timestamp.
        
        Args:
            timestamp_str: Timestamp string in ISO format
            
        Returns:
            Unix timestamp as integer
        """
        logger.debug(f"Parsing timestamp: {timestamp_str}")
        try:
            return int(time.mktime(time.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")))
        except ValueError as e:
            logger.warning(f"Invalid timestamp format: {timestamp_str}, error: {str(e)}")
            return self.get_current_timestamp()
        except Exception as e:
            logger.error(f"Error parsing timestamp: {str(e)}")
            return self.get_current_timestamp()
    
    def create_model_data(self, model_dir: Path) -> ModelData:
        """
        Create ModelData object from model directory.
        
        Args:
            model_dir: Path to the model directory
            
        Returns:
            ModelData object with model information
        """
        logger.debug(f"Creating model data for: {model_dir}")
        try:
            metadata = self.load_model_metadata(model_dir)
            
            # Get created timestamp
            created = self.get_current_timestamp()
            if "created_at" in metadata:
                created = self.parse_model_timestamp(metadata["created_at"])
            
            # Get root model name
            root = metadata.get("base_model", "unknown")
            
            logger.debug(f"Model data created for {model_dir.name}: base={root}, created={created}")
            
            return ModelData(
                id=model_dir.name,
                created=created,
                root=root
            )
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error creating model data for {model_dir}: {str(e)}\n{error_detail}")
            # Return a minimal model with default values
            return ModelData(
                id=model_dir.name,
                created=self.get_current_timestamp(),
                root="unknown"
            )
