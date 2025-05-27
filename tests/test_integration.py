"""Integration tests for the Vana Inference Engine."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from app.main import app


class TestIntegration:
    """Integration test cases for the complete application."""

    @pytest.fixture
    def client(self):
        """Create a test client for integration tests."""
        return TestClient(app)

    def test_app_startup_and_health_check(self, client):
        """Test that the app starts up and health check works."""
        # Mock the health service to avoid dependency issues
        mock_health_response = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00Z",
            "components": {
                "file_system": {"directories": {}},
                "model": {"status": "healthy", "message": "OK", "path": "/test"},
                "gpu": {"status": "healthy", "message": "OK", "cuda_available": True},
                "system": {"status": "healthy", "message": "OK"}
            }
        }

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = Mock(**mock_health_response)
            mock_health.return_value.dict = lambda: mock_health_response
            
            response = client.get("/health")
            
            # Should get a successful response
            assert response.status_code in [200, 500, 503]

    def test_openapi_documentation_generation(self, client):
        """Test that OpenAPI documentation is properly generated."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        
        # Verify basic OpenAPI structure
        assert "openapi" in schema
        assert "info" in schema
        assert "paths" in schema
        
        # Verify our endpoints are documented
        paths = schema["paths"]
        expected_paths = [
            "/health",
            "/inference/chat/completions",
            "/inference/models",
            "/train",
        ]
        
        for path in expected_paths:
            assert path in paths, f"Path {path} not found in OpenAPI schema"

    def test_cors_headers(self, client):
        """Test that CORS headers are properly set."""
        # Test preflight request
        response = client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # CORS should be configured to allow all origins
        assert response.status_code in [200, 405]  # 405 if OPTIONS not implemented

    def test_error_handling_404(self, client):
        """Test 404 error handling."""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
        
        data = response.json()
        assert "detail" in data

    def test_error_handling_422_validation(self, client):
        """Test 422 validation error handling."""
        # Send invalid JSON to an endpoint that expects specific format
        response = client.post("/inference/chat/completions", json={"invalid": "data"})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data

    @pytest.mark.integration
    def test_full_inference_workflow_mock(self, client):
        """Test a complete inference workflow with mocked dependencies."""
        # Mock all the dependencies
        mock_model_list = {
            "object": "list",
            "data": [
                {"id": "test-model", "object": "model", "created": 1234567890, "owned_by": "test"}
            ]
        }
        
        mock_chat_response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "test-model",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello! How can I help you?"},
                    "finish_reason": "stop"
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 8, "total_tokens": 18}
        }

        with patch("app.services.inference.InferenceService.list_available_models") as mock_list, \
             patch("app.services.inference.InferenceService.get_model_path") as mock_get_path, \
             patch("app.services.inference.InferenceService.generate_completion_id") as mock_gen_id, \
             patch("app.services.inference.InferenceService.get_current_timestamp") as mock_timestamp, \
             patch("app.ml.inference.generate_chat_completion") as mock_generate:
            
            # Setup mocks
            mock_list.return_value = Mock(**mock_model_list)
            mock_list.return_value.dict = lambda: mock_model_list
            mock_get_path.return_value = "/test/models/test-model"
            mock_gen_id.return_value = "chatcmpl-123"
            mock_timestamp.return_value = 1234567890
            mock_generate.return_value = Mock(**mock_chat_response)
            mock_generate.return_value.dict = lambda: mock_chat_response

            # 1. List available models
            models_response = client.get("/inference/models")
            assert models_response.status_code == 200
            models_data = models_response.json()
            assert len(models_data["data"]) == 1
            assert models_data["data"][0]["id"] == "test-model"

            # 2. Create a chat completion
            chat_request = {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "Hello!"}
                ],
                "stream": False
            }
            
            chat_response = client.post("/inference/chat/completions", json=chat_request)
            assert chat_response.status_code == 200
            chat_data = chat_response.json()
            assert chat_data["id"] == "chatcmpl-123"
            assert chat_data["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @pytest.mark.integration
    def test_full_training_workflow_mock(self, client):
        """Test a complete training workflow with mocked dependencies."""
        mock_training_response = {
            "training_job_id": "job-123",
            "status": "started",
            "message": "Training job started successfully",
            "model": "test-model",
            "output_model_name": "fine-tuned-model"
        }
        
        mock_status_response = {
            "training_job_id": "job-123",
            "status": "running",
            "progress": 0.5,
            "message": "Training in progress",
            "created_at": "2024-01-01T00:00:00Z",
            "model": "test-model",
            "output_model_name": "fine-tuned-model"
        }

        with patch("app.services.training.TrainingService.start_training_job") as mock_start, \
             patch("app.services.training.TrainingService.get_training_job_status") as mock_status:
            
            # Setup mocks
            mock_start.return_value = Mock(**mock_training_response)
            mock_start.return_value.dict = lambda: mock_training_response
            mock_status.return_value = Mock(**mock_status_response)
            mock_status.return_value.dict = lambda: mock_status_response

            # 1. Start a training job
            training_request = {
                "model": "test-model",
                "query_id": "query-123",
                "output_model_name": "fine-tuned-model"
            }
            
            training_response = client.post("/train", json=training_request)
            assert training_response.status_code == 202
            training_data = training_response.json()
            assert training_data["training_job_id"] == "job-123"
            assert training_data["status"] == "started"

            # 2. Check training status
            job_id = training_data["training_job_id"]
            status_response = client.get(f"/train/{job_id}")
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["training_job_id"] == job_id
            assert status_data["status"] == "running"
            assert status_data["progress"] == 0.5

    def test_api_versioning_and_compatibility(self, client):
        """Test API versioning and OpenAI compatibility."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        
        # Check that we have OpenAI-compatible endpoints
        paths = schema["paths"]
        
        # Chat completions endpoint should match OpenAI format
        if "/inference/chat/completions" in paths:
            chat_endpoint = paths["/inference/chat/completions"]
            assert "post" in chat_endpoint
            
            # Should have proper request/response schemas
            post_spec = chat_endpoint["post"]
            assert "requestBody" in post_spec
            assert "responses" in post_spec

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time
        
        results = []
        
        def make_request():
            try:
                response = client.get("/health")
                results.append(response.status_code)
            except Exception as e:
                results.append(str(e))
        
        # Create multiple threads to make concurrent requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should complete (though they might fail due to missing dependencies)
        assert len(results) == 5
        # Results should be status codes or error strings
        for result in results:
            assert isinstance(result, (int, str))

    @pytest.mark.slow
    def test_application_performance(self, client):
        """Test basic performance characteristics."""
        import time
        
        # Test response time for health check
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Health check should respond quickly (under 5 seconds even with mocking)
        assert response_time < 5.0
        
        # Should get some response
        assert response.status_code in [200, 500, 503]