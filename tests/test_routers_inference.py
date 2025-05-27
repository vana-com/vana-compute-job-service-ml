"""Tests for the inference router."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.models.openai import ChatCompletionRequest, ChatCompletionResponse, Choice, Message, Usage
from app.models.inference import ModelListResponse, ModelData


class TestInferenceRouter:
    """Test cases for the inference router."""

    def test_chat_completions_success(self, client: TestClient, sample_chat_messages):
        """Test successful chat completion."""
        request_data = {
            "model": "test-model",
            "messages": sample_chat_messages,
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": False
        }

        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello! I'm doing well, thank you for asking."),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=20, completion_tokens=15, total_tokens=35)
        )

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.return_value = mock_response

            response = client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            
            assert data["id"] == "chatcmpl-123"
            assert data["object"] == "chat.completion"
            assert data["model"] == "test-model"
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Hello! I'm doing well, thank you for asking."
            assert data["usage"]["total_tokens"] == 35

    def test_chat_completions_streaming(self, client: TestClient, sample_chat_messages):
        """Test streaming chat completion."""
        request_data = {
            "model": "test-model",
            "messages": sample_chat_messages,
            "temperature": 0.7,
            "max_tokens": 100,
            "stream": True
        }

        async def mock_stream_generator():
            yield b"data: {'delta': {'content': 'Hello'}}\n\n"
            yield b"data: {'delta': {'content': ' there!'}}\n\n"
            yield b"data: [DONE]\n\n"

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.return_value = mock_stream_generator()

            response = client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 200
            assert response.headers["content-type"] == "text/event-stream; charset=utf-8"

    def test_chat_completions_model_not_found(self, client: TestClient, sample_chat_messages):
        """Test chat completion with non-existent model."""
        request_data = {
            "model": "non-existent-model",
            "messages": sample_chat_messages,
            "stream": False
        }

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path:
            mock_get_path.side_effect = FileNotFoundError("Model not found")

            response = client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 404
            data = response.json()
            assert "Model 'non-existent-model' not found" in data["detail"]

    def test_chat_completions_generation_error(self, client: TestClient, sample_chat_messages):
        """Test chat completion with generation error."""
        request_data = {
            "model": "test-model",
            "messages": sample_chat_messages,
            "stream": False
        }

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.side_effect = Exception("Generation failed")

            response = client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 500
            data = response.json()
            assert "Error generating chat completion" in data["detail"]

    def test_chat_completions_invalid_request(self, client: TestClient):
        """Test chat completion with invalid request data."""
        # Missing required fields
        request_data = {
            "model": "test-model"
            # Missing messages
        }

        response = client.post("/inference/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_chat_completions_empty_messages(self, client: TestClient):
        """Test chat completion with empty messages."""
        request_data = {
            "model": "test-model",
            "messages": [],
            "stream": False
        }

        response = client.post("/inference/chat/completions", json=request_data)
        assert response.status_code == 422  # Validation error

    def test_list_models_success(self, client: TestClient):
        """Test successful model listing."""
        mock_response = ModelListResponse(
            object="list",
            data=[
                ModelData(
                    id="model-1",
                    object="model",
                    created=1234567890,
                    owned_by="test"
                ),
                ModelData(
                    id="model-2",
                    object="model",
                    created=1234567891,
                    owned_by="test"
                )
            ]
        )

        with patch("app.services.inference.InferenceService.list_available_models") as mock_list:
            mock_list.return_value = mock_response

            response = client.get("/inference/models")

            assert response.status_code == 200
            data = response.json()
            
            assert data["object"] == "list"
            assert len(data["data"]) == 2
            assert data["data"][0]["id"] == "model-1"
            assert data["data"][1]["id"] == "model-2"

    def test_list_models_error(self, client: TestClient):
        """Test model listing with error."""
        with patch("app.services.inference.InferenceService.list_available_models") as mock_list:
            mock_list.side_effect = Exception("Failed to list models")

            response = client.get("/inference/models")

            assert response.status_code == 500
            data = response.json()
            assert "Error listing models" in data["detail"]

    def test_list_models_empty(self, client: TestClient):
        """Test model listing with no models."""
        mock_response = ModelListResponse(
            object="list",
            data=[]
        )

        with patch("app.services.inference.InferenceService.list_available_models") as mock_list:
            mock_list.return_value = mock_response

            response = client.get("/inference/models")

            assert response.status_code == 200
            data = response.json()
            
            assert data["object"] == "list"
            assert len(data["data"]) == 0

    @pytest.mark.asyncio
    async def test_chat_completions_async(self, async_client, sample_chat_messages):
        """Test chat completion with async client."""
        request_data = {
            "model": "test-model",
            "messages": sample_chat_messages,
            "stream": False
        }

        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Test response"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.return_value = mock_response

            response = await async_client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "chatcmpl-123"

    def test_chat_completions_with_parameters(self, client: TestClient, sample_chat_messages):
        """Test chat completion with various parameters."""
        request_data = {
            "model": "test-model",
            "messages": sample_chat_messages,
            "temperature": 0.9,
            "top_p": 0.95,
            "max_tokens": 200,
            "stop": ["END"],
            "stream": False
        }

        mock_response = ChatCompletionResponse(
            id="chatcmpl-123",
            object="chat.completion",
            created=1234567890,
            model="test-model",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Test response"),
                    finish_reason="stop"
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        )

        with patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.return_value = mock_response

            response = client.post("/inference/chat/completions", json=request_data)

            assert response.status_code == 200
            
            # Verify that the generate function was called with the correct parameters
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args[1]["temperature"] == 0.9
            assert call_args[1]["top_p"] == 0.95
            assert call_args[1]["max_tokens"] == 200
            assert call_args[1]["stop"] == ["END"]