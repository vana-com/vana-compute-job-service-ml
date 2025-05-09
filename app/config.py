import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Base directories
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = Path(os.getenv("OUTPUT_PATH", BASE_DIR / "data/output"))
WORKING_DIR = Path(os.getenv("WORKING_PATH", BASE_DIR / "data/working"))

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
WORKING_DIR.mkdir(exist_ok=True, parents=True)

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Vana Inference Engine"
        
    # Model settings
    MODEL_DIR: Path = WORKING_DIR / "models"
    DEFAULT_BASE_MODEL: str = "meta-llama/Llama-2-7b-hf"
    
    # Training settings
    MAX_SEQ_LENGTH: int = 512
    BATCH_SIZE: int = 4
    LEARNING_RATE: float = 2e-4
    NUM_EPOCHS: int = 3
    
    # Inference settings
    MAX_NEW_TOKENS: int = 512
    TEMPERATURE: float = 0.7
    TOP_P: float = 0.9
    
    # Ensure model directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MODEL_DIR.mkdir(exist_ok=True, parents=True)

# Create global settings object
settings = Settings()