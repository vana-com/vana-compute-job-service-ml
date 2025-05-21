from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict, Literal
from app.config import settings

class QueryParams(BaseModel):
    """Parameters for a query to the database."""
    compute_job_id: int
    refiner_id: int
    query: str
    query_signature: str
    params: Optional[List[Any]] = None

    class Config:
        json_schema_extra = {
            "example": {
                "compute_job_id": 123,
                "refiner_id": 456,
                "query": "SELECT * FROM data WHERE user_id = ? LIMIT 100",
                "query_signature": "<signed query contents>",
                "params": ["user123"]
            }
        }

class TrainingParameters(BaseModel):
    """Training hyperparameters and configuration."""
    num_epochs: int = Field(default=3, gt=0, description="Number of training epochs")
    learning_rate: float = Field(default=2e-4, gt=0, description="Learning rate for optimizer")
    batch_size: int = Field(default=4, gt=0, description="Batch size for training")
    max_seq_length: int = Field(default=512, gt=0, description="Maximum sequence length")
    
    class Config:
        json_schema_extra = {
            "example": {
                "num_epochs": 3,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "max_seq_length": 512
            }
        }

class TrainingRequest(BaseModel):
    """Training request model."""
    model_name: str = Field(
        default=settings.DEFAULT_BASE_MODEL,
        description="Name or path of the base model to fine-tune"
    )
    output_dir: Optional[str] = Field(
        default=None, 
        description="Directory name to save the fine-tuned model"
    )
    training_params: Optional[TrainingParameters] = Field(
        default=None,
        description="Hyperparameters for training"
    )
    query_id: Optional[str] = Field(
        default=None,
        description="ID of an existing query to use for training data"
    )
    query_params: Optional[QueryParams] = Field(
        default=None,
        description="Parameters for a new query to generate training data"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "meta-llama/Llama-2-7b-hf",
                "output_dir": "my_finetuned_model",
                "training_params": {
                    "num_epochs": 3,
                    "learning_rate": 2e-4,
                    "batch_size": 4,
                    "max_seq_length": 512
                },
                "query_params": {
                    "compute_job_id": 123,
                    "refiner_id": 456,
                    "query": "SELECT * FROM data WHERE user_id = ? LIMIT 100",
                    "query_signature": "<signed query contents>",
                    "params": ["user123"]
                }
            }
        }

class TrainingStatus(BaseModel):
    """Model representing the status of a training job."""
    status: Literal["pending", "started", "running", "completed", "failed"]
    progress: Optional[float] = Field(default=None, ge=0, le=100, description="Progress percentage (0-100)")
    current_epoch: Optional[int] = Field(default=None, description="Current training epoch")
    total_epochs: Optional[int] = Field(default=None, description="Total number of epochs")
    loss: Optional[float] = Field(default=None, description="Current training loss")
    model_path: Optional[str] = Field(default=None, description="Path to the trained model")
    error_message: Optional[str] = Field(default=None, description="Error message if training failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "running",
                "progress": 45.2,
                "current_epoch": 2,
                "total_epochs": 5,
                "loss": 0.342
            }
        }

class TrainingResponse(BaseModel):
    """Response model for training job creation."""
    job_id: str = Field(..., description="Unique identifier for the training job")
    query_id: str = Field(..., description="ID of the query used for training data")
    status: Literal["pending", "started", "failed"] = Field(
        ..., 
        description="Initial status of the training job"
    )
    message: str = Field(..., description="Human-readable message about the job")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "train_a1b2c3d4",
                "query_id": "query_e5f6g7h8",
                "status": "started",
                "message": "Training job started. Connect to /train/train_a1b2c3d4/events for real-time updates."
            }
        }

class TrainingEvent(BaseModel):
    """Model representing an event from the training process."""
    type: Literal["progress", "log", "complete", "error"] = Field(
        ..., 
        description="Type of the training event"
    )
    data: Dict[str, Any] = Field(..., description="Event data payload")
    timestamp: int = Field(..., description="Event timestamp (Unix epoch time)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "progress",
                "data": {
                    "epoch": 2,
                    "progress": 45.2,
                    "loss": 0.342
                },
                "timestamp": 1617293424
            }
        }