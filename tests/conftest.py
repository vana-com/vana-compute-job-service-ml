"""Pytest configuration and fixtures for Vana Inference Engine tests."""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.config import settings


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client() -> TestClient:
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Create an async test client for the FastAPI app."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("app.config.settings") as mock:
        mock.WORKING_DIR = settings.WORKING_DIR
        mock.OUTPUT_DIR = settings.OUTPUT_DIR
        mock.MODEL_DIR = settings.MODEL_DIR
        mock.DEFAULT_BASE_MODEL = "test-model"
        mock.MAX_SEQ_LENGTH = 256
        mock.BATCH_SIZE = 2
        mock.LEARNING_RATE = 1e-4
        mock.NUM_EPOCHS = 1
        mock.MAX_NEW_TOKENS = 100
        mock.TEMPERATURE = 0.5
        mock.TOP_P = 0.8
        yield mock


@pytest.fixture
def mock_model_path(tmp_path):
    """Create a temporary model path for testing."""
    model_dir = tmp_path / "models" / "test-model"
    model_dir.mkdir(parents=True)
    return model_dir


@pytest.fixture
def mock_health_service():
    """Mock health service for testing."""
    with patch("app.services.health.HealthService") as mock:
        yield mock


@pytest.fixture
def mock_inference_service():
    """Mock inference service for testing."""
    with patch("app.services.inference.InferenceService") as mock:
        yield mock


@pytest.fixture
def mock_training_service():
    """Mock training service for testing."""
    with patch("app.services.training.TrainingService") as mock:
        yield mock


@pytest.fixture
def sample_chat_messages():
    """Sample chat messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"}
    ]


@pytest.fixture
def sample_training_request():
    """Sample training request for testing."""
    return {
        "model": "test-model",
        "query_id": "test-query-123",
        "output_model_name": "fine-tuned-model",
        "max_seq_length": 256,
        "batch_size": 2,
        "learning_rate": 1e-4,
        "num_epochs": 1
    }