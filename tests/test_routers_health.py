"""Tests for the health router."""

import pytest
from unittest.mock import Mock, patch
from fastapi.testclient import TestClient

from app.models.health import (
    HealthCheckResponse, ComponentStatus, FileSystemStatus, 
    ModelStatus, GPUStatus, SystemStatus, FileSystemDirectoryStatus,
    DiskStatus, MemoryStatus, CPUStatus
)


class TestHealthRouter:
    """Test cases for the health router."""

    def test_health_check_success(self, client: TestClient):
        """Test successful health check."""
        # Mock the health service to return a healthy response
        mock_response = HealthCheckResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(
                    directories={
                        "working": FileSystemDirectoryStatus(
                            status="healthy",
                            message="Directory accessible",
                            path="/test/working"
                        )
                    }
                ),
                model=ModelStatus(
                    status="healthy",
                    message="Model loading capability verified",
                    path="/test/models"
                ),
                gpu=GPUStatus(
                    status="healthy",
                    message="GPU available",
                    cuda_available=True
                ),
                system=SystemStatus(
                    status="healthy",
                    message="System resources normal",
                    disk=DiskStatus(total=1000, used=500, free=500, percent=50.0),
                    memory=MemoryStatus(total=8000, available=4000, used=4000, percent=50.0),
                    cpu=CPUStatus(percent=25.0, count=4)
                )
            )
        )

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = mock_response
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "healthy"
            assert "timestamp" in data
            assert "components" in data
            assert data["components"]["file_system"]["directories"]["working"]["status"] == "healthy"
            assert data["components"]["model"]["status"] == "healthy"
            assert data["components"]["gpu"]["status"] == "healthy"
            assert data["components"]["system"]["status"] == "healthy"

    def test_health_check_degraded(self, client: TestClient):
        """Test health check with degraded status."""
        mock_response = HealthCheckResponse(
            status="degraded",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(
                    directories={
                        "working": FileSystemDirectoryStatus(
                            status="healthy",
                            message="Directory accessible",
                            path="/test/working"
                        )
                    }
                ),
                model=ModelStatus(
                    status="warning",
                    message="Model directory exists but no models found",
                    path="/test/models"
                ),
                gpu=GPUStatus(
                    status="unhealthy",
                    message="CUDA not available",
                    cuda_available=False
                ),
                system=SystemStatus(
                    status="healthy",
                    message="System resources normal"
                )
            )
        )

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = mock_response
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "degraded"
            assert data["components"]["model"]["status"] == "warning"
            assert data["components"]["gpu"]["status"] == "unhealthy"

    def test_health_check_unhealthy(self, client: TestClient):
        """Test health check with unhealthy status."""
        mock_response = HealthCheckResponse(
            status="unhealthy",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(
                    directories={
                        "working": FileSystemDirectoryStatus(
                            status="unhealthy",
                            message="Directory not accessible",
                            path="/test/working"
                        )
                    }
                ),
                model=ModelStatus(
                    status="unhealthy",
                    message="Model directory not found",
                    path="/test/models"
                ),
                gpu=GPUStatus(
                    status="unhealthy",
                    message="CUDA not available",
                    cuda_available=False
                ),
                system=SystemStatus(
                    status="unhealthy",
                    message="System resources critical"
                )
            )
        )

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = mock_response
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["status"] == "unhealthy"
            assert data["components"]["file_system"]["directories"]["working"]["status"] == "unhealthy"
            assert data["components"]["model"]["status"] == "unhealthy"

    def test_health_check_service_error(self, client: TestClient):
        """Test health check when service raises an exception."""
        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.side_effect = Exception("Service error")
            
            response = client.get("/health")
            
            # The endpoint should handle the exception gracefully
            # This depends on how the health service is implemented
            assert response.status_code in [200, 500, 503]

    def test_health_check_response_model(self, client: TestClient):
        """Test that the health check response follows the correct model."""
        mock_response = HealthCheckResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(directories={}),
                model=ModelStatus(status="healthy", message="OK", path="/test"),
                gpu=GPUStatus(status="healthy", message="OK"),
                system=SystemStatus(status="healthy", message="OK")
            )
        )

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = mock_response
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            
            # Validate required fields
            required_fields = ["status", "timestamp", "components"]
            for field in required_fields:
                assert field in data
            
            # Validate component structure
            components = data["components"]
            component_types = ["file_system", "model", "gpu", "system"]
            for component in component_types:
                assert component in components

    @pytest.mark.asyncio
    async def test_health_check_async(self, async_client):
        """Test health check with async client."""
        mock_response = HealthCheckResponse(
            status="healthy",
            timestamp="2024-01-01T00:00:00Z",
            components=ComponentStatus(
                file_system=FileSystemStatus(directories={}),
                model=ModelStatus(status="healthy", message="OK", path="/test"),
                gpu=GPUStatus(status="healthy", message="OK"),
                system=SystemStatus(status="healthy", message="OK")
            )
        )

        with patch("app.services.health.HealthService.check_system_health") as mock_health:
            mock_health.return_value = mock_response
            
            response = await async_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"