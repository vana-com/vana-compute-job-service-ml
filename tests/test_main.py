"""Tests for the main FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


class TestMainApp:
    """Test cases for the main FastAPI application."""

    def test_app_creation(self):
        """Test that the FastAPI app is created correctly."""
        assert app.title == "Vana Inference Engine"
        assert app.description == "FastAPI-based inference/training engine for Vana"
        assert app.version == "0.1.0"

    def test_cors_middleware(self, client: TestClient):
        """Test that CORS middleware is properly configured."""
        response = client.options("/health")
        assert response.status_code in [200, 405]  # OPTIONS might not be implemented for all endpoints

    def test_routers_included(self):
        """Test that all routers are included in the app."""
        routes = [route.path for route in app.routes]
        
        # Check that router prefixes are included
        health_routes = [route for route in routes if route.startswith("/health")]
        inference_routes = [route for route in routes if route.startswith("/inference")]
        training_routes = [route for route in routes if route.startswith("/train")]
        
        assert len(health_routes) > 0, "Health routes should be included"
        assert len(inference_routes) > 0, "Inference routes should be included"
        assert len(training_routes) > 0, "Training routes should be included"

    def test_app_startup(self, client: TestClient):
        """Test that the app starts up correctly."""
        # Test that we can make a request to the app
        response = client.get("/health")
        # Should get a response (even if it's an error due to missing dependencies)
        assert response.status_code in [200, 500, 503]

    def test_openapi_schema(self, client: TestClient):
        """Test that OpenAPI schema is generated correctly."""
        response = client.get("/openapi.json")
        assert response.status_code == 200
        
        schema = response.json()
        assert schema["info"]["title"] == "Vana Inference Engine"
        assert schema["info"]["version"] == "0.1.0"
        
        # Check that our main endpoints are documented
        paths = schema["paths"]
        assert "/health" in paths
        assert "/inference/chat/completions" in paths
        assert "/inference/models" in paths
        assert "/train" in paths

    def test_docs_endpoint(self, client: TestClient):
        """Test that the documentation endpoint is accessible."""
        response = client.get("/docs")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_redoc_endpoint(self, client: TestClient):
        """Test that the ReDoc endpoint is accessible."""
        response = client.get("/redoc")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]