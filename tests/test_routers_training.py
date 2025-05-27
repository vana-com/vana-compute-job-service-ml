"""Tests for the training router."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.models.training import (
    TrainingRequest, TrainingResponse, TrainingStatus, TrainingEvent
)


class TestTrainingRouter:
    """Test cases for the training router."""

    def test_start_training_success(self, client: TestClient, sample_training_request):
        """Test successful training job start."""
        mock_response = TrainingResponse(
            training_job_id="job-123",
            status="started",
            message="Training job started successfully",
            model=sample_training_request["model"],
            output_model_name=sample_training_request["output_model_name"]
        )

        with patch("app.services.training.TrainingService.start_training_job") as mock_start:
            mock_start.return_value = mock_response

            response = client.post("/train", json=sample_training_request)

            assert response.status_code == 202
            data = response.json()
            
            assert data["training_job_id"] == "job-123"
            assert data["status"] == "started"
            assert data["model"] == sample_training_request["model"]
            assert data["output_model_name"] == sample_training_request["output_model_name"]

    def test_start_training_invalid_request(self, client: TestClient):
        """Test training start with invalid request."""
        # Missing required fields
        invalid_request = {
            "model": "test-model"
            # Missing other required fields
        }

        response = client.post("/train", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_start_training_service_error(self, client: TestClient, sample_training_request):
        """Test training start with service error."""
        with patch("app.services.training.TrainingService.start_training_job") as mock_start:
            mock_start.side_effect = HTTPException(status_code=400, detail="Invalid query_id")

            response = client.post("/train", json=sample_training_request)

            assert response.status_code == 400
            data = response.json()
            assert "Invalid query_id" in data["detail"]

    def test_start_training_with_query_params(self, client: TestClient):
        """Test training start with query parameters instead of query_id."""
        request_data = {
            "model": "test-model",
            "query_params": {
                "table": "test_table",
                "columns": ["col1", "col2"],
                "limit": 1000
            },
            "output_model_name": "fine-tuned-model"
        }

        mock_response = TrainingResponse(
            training_job_id="job-456",
            status="started",
            message="Training job started successfully",
            model=request_data["model"],
            output_model_name=request_data["output_model_name"]
        )

        with patch("app.services.training.TrainingService.start_training_job") as mock_start:
            mock_start.return_value = mock_response

            response = client.post("/train", json=request_data)

            assert response.status_code == 202
            data = response.json()
            assert data["training_job_id"] == "job-456"

    def test_get_training_status_success(self, client: TestClient):
        """Test successful training status retrieval."""
        job_id = "job-123"
        mock_status = TrainingStatus(
            training_job_id=job_id,
            status="running",
            progress=0.5,
            message="Training in progress",
            created_at="2024-01-01T00:00:00Z",
            started_at="2024-01-01T00:01:00Z",
            model="test-model",
            output_model_name="fine-tuned-model"
        )

        with patch("app.services.training.TrainingService.get_training_job_status") as mock_get_status:
            mock_get_status.return_value = mock_status

            response = client.get(f"/train/{job_id}")

            assert response.status_code == 200
            data = response.json()
            
            assert data["training_job_id"] == job_id
            assert data["status"] == "running"
            assert data["progress"] == 0.5
            assert data["model"] == "test-model"

    def test_get_training_status_not_found(self, client: TestClient):
        """Test training status retrieval for non-existent job."""
        job_id = "non-existent-job"

        with patch("app.services.training.TrainingService.get_training_job_status") as mock_get_status:
            mock_get_status.side_effect = HTTPException(status_code=404, detail="Training job not found")

            response = client.get(f"/train/{job_id}")

            assert response.status_code == 404
            data = response.json()
            assert "Training job not found" in data["detail"]

    def test_stream_training_events_success(self, client: TestClient):
        """Test successful training events streaming."""
        job_id = "job-123"

        # Mock the status file existence check
        with patch("app.config.settings.WORKING_DIR") as mock_working_dir:
            mock_status_file = Mock()
            mock_status_file.exists.return_value = True
            mock_working_dir.__truediv__.return_value = mock_status_file

            with patch("app.services.training.generate_sse_events") as mock_generate_sse:
                async def mock_event_generator():
                    yield "data: {'event': 'progress', 'data': {'progress': 0.1}}\n\n"
                    yield "data: {'event': 'log', 'data': {'message': 'Training started'}}\n\n"

                mock_generate_sse.return_value = mock_event_generator()

                response = client.get(f"/train/{job_id}/events")

                assert response.status_code == 200
                assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
                assert "Cache-Control" in response.headers
                assert response.headers["Cache-Control"] == "no-cache"

    def test_stream_training_events_job_not_found(self, client: TestClient):
        """Test training events streaming for non-existent job."""
        job_id = "non-existent-job"

        # Mock the status file existence check to return False
        with patch("app.config.settings.WORKING_DIR") as mock_working_dir:
            mock_status_file = Mock()
            mock_status_file.exists.return_value = False
            mock_working_dir.__truediv__.return_value = mock_status_file

            response = client.get(f"/train/{job_id}/events")

            assert response.status_code == 404
            data = response.json()
            assert f"Training job {job_id} not found" in data["detail"]

    def test_get_training_event_history_success(self, client: TestClient):
        """Test successful training event history retrieval."""
        job_id = "job-123"
        mock_events = [
            TrainingEvent(
                event="progress",
                timestamp="2024-01-01T00:00:00Z",
                data={"progress": 0.1}
            ),
            TrainingEvent(
                event="log",
                timestamp="2024-01-01T00:01:00Z",
                data={"message": "Training started"}
            ),
            TrainingEvent(
                event="complete",
                timestamp="2024-01-01T01:00:00Z",
                data={"final_loss": 0.05}
            )
        ]

        # Mock the status file existence check
        with patch("app.config.settings.WORKING_DIR") as mock_working_dir:
            mock_status_file = Mock()
            mock_status_file.exists.return_value = True
            mock_working_dir.__truediv__.return_value = mock_status_file

            with patch("app.services.training.TrainingService.get_training_events") as mock_get_events:
                mock_get_events.return_value = mock_events

                response = client.get(f"/train/{job_id}/events/history")

                assert response.status_code == 200
                data = response.json()
                
                assert len(data) == 3
                assert data[0]["event"] == "progress"
                assert data[1]["event"] == "log"
                assert data[2]["event"] == "complete"
                assert data[0]["data"]["progress"] == 0.1
                assert data[2]["data"]["final_loss"] == 0.05

    def test_get_training_event_history_job_not_found(self, client: TestClient):
        """Test training event history retrieval for non-existent job."""
        job_id = "non-existent-job"

        # Mock the status file existence check to return False
        with patch("app.config.settings.WORKING_DIR") as mock_working_dir:
            mock_status_file = Mock()
            mock_status_file.exists.return_value = False
            mock_working_dir.__truediv__.return_value = mock_status_file

            response = client.get(f"/train/{job_id}/events/history")

            assert response.status_code == 404
            data = response.json()
            assert f"Training job {job_id} not found" in data["detail"]

    def test_get_training_event_history_empty(self, client: TestClient):
        """Test training event history retrieval with no events."""
        job_id = "job-123"

        # Mock the status file existence check
        with patch("app.config.settings.WORKING_DIR") as mock_working_dir:
            mock_status_file = Mock()
            mock_status_file.exists.return_value = True
            mock_working_dir.__truediv__.return_value = mock_status_file

            with patch("app.services.training.TrainingService.get_training_events") as mock_get_events:
                mock_get_events.return_value = []

                response = client.get(f"/train/{job_id}/events/history")

                assert response.status_code == 200
                data = response.json()
                assert len(data) == 0

    @pytest.mark.asyncio
    async def test_start_training_async(self, async_client, sample_training_request):
        """Test training start with async client."""
        mock_response = TrainingResponse(
            training_job_id="job-123",
            status="started",
            message="Training job started successfully",
            model=sample_training_request["model"],
            output_model_name=sample_training_request["output_model_name"]
        )

        with patch("app.services.training.TrainingService.start_training_job") as mock_start:
            mock_start.return_value = mock_response

            response = await async_client.post("/train", json=sample_training_request)

            assert response.status_code == 202
            data = response.json()
            assert data["training_job_id"] == "job-123"

    def test_training_with_custom_parameters(self, client: TestClient):
        """Test training with custom hyperparameters."""
        request_data = {
            "model": "test-model",
            "query_id": "test-query-123",
            "output_model_name": "custom-fine-tuned-model",
            "max_seq_length": 1024,
            "batch_size": 8,
            "learning_rate": 1e-5,
            "num_epochs": 5
        }

        mock_response = TrainingResponse(
            training_job_id="job-custom",
            status="started",
            message="Training job started with custom parameters",
            model=request_data["model"],
            output_model_name=request_data["output_model_name"]
        )

        with patch("app.services.training.TrainingService.start_training_job") as mock_start:
            mock_start.return_value = mock_response

            response = client.post("/train", json=request_data)

            assert response.status_code == 202
            data = response.json()
            assert data["training_job_id"] == "job-custom"
            
            # Verify the service was called with the custom parameters
            mock_start.assert_called_once()
            call_args = mock_start.call_args[0][0]  # First argument (TrainingRequest)
            assert call_args.max_seq_length == 1024
            assert call_args.batch_size == 8
            assert call_args.learning_rate == 1e-5
            assert call_args.num_epochs == 5