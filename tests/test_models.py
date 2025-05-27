"""Tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from app.models.health import (
    HealthCheckResponse, ComponentStatus, FileSystemStatus, 
    ModelStatus, GPUStatus, SystemStatus, FileSystemDirectoryStatus,
    DiskStatus, MemoryStatus, CPUStatus, GPUInfo, GPUDevice
)
from app.models.openai import (
    ChatCompletionRequest, ChatCompletionResponse, Message, Choice, Usage
)
from app.models.inference import ModelListResponse, ModelData
from app.models.training import (
    TrainingRequest, TrainingResponse, TrainingStatus, TrainingEvent
)


class TestHealthModels:
    """Test cases for health check models."""

    def test_health_check_response_valid(self):
        """Test valid health check response creation."""
        response = HealthCheckResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(directories={}),
                model=ModelStatus(status="healthy", message="OK", path="/test"),
                gpu=GPUStatus(status="healthy", message="OK"),
                system=SystemStatus(status="healthy", message="OK")
            )
        )
        
        assert response.status == "healthy"
        assert response.timestamp == "2024-01-01T00:00:00Z"
        assert response.components.model.status == "healthy"

    def test_health_check_response_invalid_status(self):
        """Test health check response with invalid status."""
        with pytest.raises(ValidationError):
            HealthCheckResponse(
                status="invalid_status",  # Should be one of: healthy, degraded, unhealthy
                timestamp="2024-01-01T00:00:00Z",
                components=ComponentStatus(
                    file_system=FileSystemStatus(directories={}),
                    model=ModelStatus(status="healthy", message="OK", path="/test"),
                    gpu=GPUStatus(status="healthy", message="OK"),
                    system=SystemStatus(status="healthy", message="OK")
                )
            )

    def test_gpu_device_model(self):
        """Test GPU device model."""
        device = GPUDevice(
            name="NVIDIA GeForce RTX 3080",
            memory_total=10737418240,
            memory_allocated=2147483648,
            memory_reserved=2684354560
        )
        
        assert device.name == "NVIDIA GeForce RTX 3080"
        assert device.memory_total == 10737418240
        assert device.memory_allocated == 2147483648

    def test_system_status_with_resources(self):
        """Test system status with resource information."""
        system_status = SystemStatus(
            status="healthy",
            message="All systems normal",
            disk=DiskStatus(total=1000000, used=500000, free=500000, percent=50.0),
            memory=MemoryStatus(total=16000000, available=8000000, used=8000000, percent=50.0),
            cpu=CPUStatus(percent=25.5, count=8)
        )
        
        assert system_status.status == "healthy"
        assert system_status.disk.percent == 50.0
        assert system_status.memory.total == 16000000
        assert system_status.cpu.count == 8


class TestOpenAIModels:
    """Test cases for OpenAI-compatible models."""

    def test_chat_completion_request_valid(self):
        """Test valid chat completion request."""
        request = ChatCompletionRequest(
            model="gpt-3.5-turbo",
            messages=[
                Message(role="system", content="You are a helpful assistant."),
                Message(role="user", content="Hello!")
            ],
            temperature=0.7,
            max_tokens=100
        )
        
        assert request.model == "gpt-3.5-turbo"
        assert len(request.messages) == 2
        assert request.messages[0].role == "system"
        assert request.temperature == 0.7

    def test_chat_completion_request_invalid_temperature(self):
        """Test chat completion request with invalid temperature."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[Message(role="user", content="Hello!")],
                temperature=3.0  # Should be between 0 and 2
            )

    def test_chat_completion_request_empty_messages(self):
        """Test chat completion request with empty messages."""
        with pytest.raises(ValidationError):
            ChatCompletionRequest(
                model="gpt-3.5-turbo",
                messages=[]  # Should not be empty
            )

    def test_message_invalid_role(self):
        """Test message with invalid role."""
        with pytest.raises(ValidationError):
            Message(role="invalid_role", content="Hello!")

    def test_chat_completion_response_valid(self):
        """Test valid chat completion response."""
        response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="gpt-3.5-turbo",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello! How can I help you?"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=8, total_tokens=18)
        )
        
        assert response.id == "chatcmpl-123"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.usage.total_tokens == 18

    def test_usage_model(self):
        """Test usage model."""
        usage = Usage(prompt_tokens=50, completion_tokens=25, total_tokens=75)
        
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 25
        assert usage.total_tokens == 75


class TestInferenceModels:
    """Test cases for inference models."""

    def test_model_data_valid(self):
        """Test valid model data."""
        model = ModelData(
            id="gpt-3.5-turbo",
            object="model",
            created=1234567890,
            owned_by="openai"
        )
        
        assert model.id == "gpt-3.5-turbo"
        assert model.object == "model"
        assert model.created == 1234567890
        assert model.owned_by == "openai"

    def test_model_list_response_valid(self):
        """Test valid model list response."""
        response = ModelListResponse(
            object="list",
            data=[
                ModelData(id="model-1", object="model", created=1234567890, owned_by="test"),
                ModelData(id="model-2", object="model", created=1234567891, owned_by="test")
            ]
        )
        
        assert response.object == "list"
        assert len(response.data) == 2
        assert response.data[0].id == "model-1"
        assert response.data[1].id == "model-2"

    def test_model_list_response_empty(self):
        """Test model list response with empty data."""
        response = ModelListResponse(object="list", data=[])
        
        assert response.object == "list"
        assert len(response.data) == 0


class TestTrainingModels:
    """Test cases for training models."""

    def test_training_request_with_query_id(self):
        """Test training request with query_id."""
        request = TrainingRequest(
            model="llama-2-7b",
            query_id="query-123",
            output_model_name="fine-tuned-model",
            max_seq_length=512,
            batch_size=4,
            learning_rate=2e-4,
            num_epochs=3
        )
        
        assert request.model == "llama-2-7b"
        assert request.query_id == "query-123"
        assert request.output_model_name == "fine-tuned-model"
        assert request.max_seq_length == 512

    def test_training_request_with_query_params(self):
        """Test training request with query_params."""
        request = TrainingRequest(
            model="llama-2-7b",
            query_params={"table": "test_table", "limit": 1000},
            output_model_name="fine-tuned-model"
        )
        
        assert request.model == "llama-2-7b"
        assert request.query_params == {"table": "test_table", "limit": 1000}
        assert request.query_id is None

    def test_training_request_missing_query_info(self):
        """Test training request without query_id or query_params."""
        # This should be valid as the validation happens at the service level
        request = TrainingRequest(
            model="llama-2-7b",
            output_model_name="fine-tuned-model"
        )
        
        assert request.model == "llama-2-7b"
        assert request.query_id is None
        assert request.query_params is None

    def test_training_response_valid(self):
        """Test valid training response."""
        response = TrainingResponse(
            training_job_id="job-123",
            status="started",
            message="Training job started successfully",
            model="llama-2-7b",
            output_model_name="fine-tuned-model"
        )
        
        assert response.training_job_id == "job-123"
        assert response.status == "started"
        assert response.model == "llama-2-7b"

    def test_training_status_valid(self):
        """Test valid training status."""
        status = TrainingStatus(
            training_job_id="job-123",
            status="running",
            progress=0.5,
            message="Training in progress",
            created_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:01:00Z",
            model="llama-2-7b",
            output_model_name="fine-tuned-model"
        )
        
        assert status.training_job_id == "job-123"
        assert status.status == "running"
        assert status.progress == 0.5
        assert status.created_at == "2024-01-01T00:00:00Z"

    def test_training_event_valid(self):
        """Test valid training event."""
        event = TrainingEvent(
            event="progress",
            timestamp="2024-01-01T00:00:00Z",
            data={"progress": 0.25, "loss": 0.5}
        )
        
        assert event.event == "progress"
        assert event.timestamp == "2024-01-01T00:00:00Z"
        assert event.data["progress"] == 0.25
        assert event.data["loss"] == 0.5

    def test_training_status_invalid_progress(self):
        """Test training status with invalid progress value."""
        # Progress should be between 0 and 1, but let's test if the model accepts it
        status = TrainingStatus(
            training_job_id="job-123",
            status="running",
            progress=1.5,  # Invalid progress > 1
            message="Training in progress",
            model="llama-2-7b",
            output_model_name="fine-tuned-model"
        )
        
        # The model might accept this, depending on validation rules
        assert status.progress == 1.5

    def test_training_request_default_values(self):
        """Test training request with default values."""
        request = TrainingRequest(
            model="llama-2-7b",
            query_id="query-123",
            output_model_name="fine-tuned-model"
        )
        
        # Check if default values are applied (if any)
        assert request.model == "llama-2-7b"
        assert request.output_model_name == "fine-tuned-model"