import uuid
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

from fastapi import HTTPException, status
from app.models.inference import ModelData, ModelListResponse

class InferenceService:
    """Service for model inference and model management."""
    
    def __init__(self, settings=None):
        """
        Initialize the inference service.
        
        Args:
            settings: Application settings, defaults to global settings
        """
        from app.config import settings as global_settings
        self.settings = settings or global_settings
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to a model directory."""
        model_path = self.settings.OUTPUT_DIR / model_name
        if not model_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model not found: {model_name}"
            )
        return model_path
    
    def generate_completion_id(self) -> str:
        """Generate a unique completion ID."""
        return f"chatcmpl-{uuid.uuid4().hex}"
    
    def get_current_timestamp(self) -> int:
        """Get current Unix timestamp."""
        return int(time.time())
    
    def list_available_models(self) -> ModelListResponse:
        """
        List all available models.
        
        Returns:
            ModelListResponse containing available models
        """
        models_data: List[ModelData] = []
        
        for model_dir in self.settings.OUTPUT_DIR.glob("*"):
            if model_dir.is_dir() and (model_dir / "config.json").exists():
                models_data.append(self.create_model_data(model_dir))
        
        return ModelListResponse(data=models_data)
    
    def load_model_metadata(self, model_dir: Path) -> Dict[str, Any]:
        """Load model metadata from file."""
        metadata_file = model_dir / "metadata.json"
        if not metadata_file.exists():
            return {}
        
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    
    def parse_model_timestamp(self, timestamp_str: str) -> int:
        """Parse timestamp string to Unix timestamp."""
        try:
            return int(time.mktime(time.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")))
        except Exception:
            return self.get_current_timestamp()
    
    def create_model_data(self, model_dir: Path) -> ModelData:
        """Create ModelData object from model directory."""
        metadata = self.load_model_metadata(model_dir)
        
        # Get created timestamp
        created = self.get_current_timestamp()
        if "created_at" in metadata:
            created = self.parse_model_timestamp(metadata["created_at"])
        
        # Get root model name
        root = metadata.get("base_model", "unknown")
        
        return ModelData(
            id=model_dir.name,
            created=created,
            root=root
        )
