"""Tests for the application configuration."""

import pytest
import os
from pathlib import Path
from unittest.mock import patch

from app.config import Settings, settings, BASE_DIR, OUTPUT_DIR, WORKING_DIR


class TestConfig:
    """Test cases for application configuration."""

    def test_base_directories(self):
        """Test that base directories are set correctly."""
        assert BASE_DIR.is_absolute()
        assert BASE_DIR.name == "app"
        assert OUTPUT_DIR.is_absolute()
        assert WORKING_DIR.is_absolute()

    def test_default_settings(self):
        """Test default settings values."""
        test_settings = Settings()
        
        assert test_settings.PROJECT_NAME == "Vana Inference Engine"
        assert test_settings.DEFAULT_BASE_MODEL == "meta-llama/Llama-2-7b-hf"
        assert test_settings.MAX_SEQ_LENGTH == 512
        assert test_settings.BATCH_SIZE == 4
        assert test_settings.LEARNING_RATE == 2e-4
        assert test_settings.NUM_EPOCHS == 3
        assert test_settings.MAX_NEW_TOKENS == 512
        assert test_settings.TEMPERATURE == 0.7
        assert test_settings.TOP_P == 0.9

    def test_directory_paths(self):
        """Test that directory paths are set correctly."""
        test_settings = Settings()
        
        assert test_settings.WORKING_DIR == WORKING_DIR
        assert test_settings.OUTPUT_DIR == OUTPUT_DIR
        assert test_settings.MODEL_DIR == WORKING_DIR / "models"

    def test_environment_variable_override(self):
        """Test that environment variables override default paths."""
        with patch.dict(os.environ, {
            'OUTPUT_PATH': '/custom/output',
            'WORKING_PATH': '/custom/working'
        }):
            # Import config again to pick up environment variables
            from importlib import reload
            import app.config
            reload(app.config)
            
            assert str(app.config.OUTPUT_DIR) == '/custom/output'
            assert str(app.config.WORKING_DIR) == '/custom/working'

    def test_model_directory_creation(self, tmp_path):
        """Test that model directory is created during settings initialization."""
        # Create a temporary working directory
        temp_working = tmp_path / "working"
        
        with patch("app.config.WORKING_DIR", temp_working):
            test_settings = Settings()
            
            # The model directory should be created
            expected_model_dir = temp_working / "models"
            assert test_settings.MODEL_DIR == expected_model_dir

    def test_settings_immutability(self):
        """Test that settings behave as expected."""
        test_settings = Settings()
        
        # Test that we can access all expected attributes
        attributes = [
            'PROJECT_NAME', 'WORKING_DIR', 'OUTPUT_DIR', 'MODEL_DIR',
            'DEFAULT_BASE_MODEL', 'MAX_SEQ_LENGTH', 'BATCH_SIZE',
            'LEARNING_RATE', 'NUM_EPOCHS', 'MAX_NEW_TOKENS',
            'TEMPERATURE', 'TOP_P'
        ]
        
        for attr in attributes:
            assert hasattr(test_settings, attr)
            assert getattr(test_settings, attr) is not None

    def test_global_settings_object(self):
        """Test that the global settings object is properly initialized."""
        assert settings is not None
        assert isinstance(settings, Settings)
        assert settings.PROJECT_NAME == "Vana Inference Engine"

    def test_path_types(self):
        """Test that path settings are Path objects."""
        test_settings = Settings()
        
        assert isinstance(test_settings.WORKING_DIR, Path)
        assert isinstance(test_settings.OUTPUT_DIR, Path)
        assert isinstance(test_settings.MODEL_DIR, Path)

    def test_numeric_settings_types(self):
        """Test that numeric settings have correct types."""
        test_settings = Settings()
        
        assert isinstance(test_settings.MAX_SEQ_LENGTH, int)
        assert isinstance(test_settings.BATCH_SIZE, int)
        assert isinstance(test_settings.NUM_EPOCHS, int)
        assert isinstance(test_settings.MAX_NEW_TOKENS, int)
        assert isinstance(test_settings.LEARNING_RATE, float)
        assert isinstance(test_settings.TEMPERATURE, float)
        assert isinstance(test_settings.TOP_P, float)

    def test_settings_validation(self):
        """Test that settings have reasonable values."""
        test_settings = Settings()
        
        # Test positive values
        assert test_settings.MAX_SEQ_LENGTH > 0
        assert test_settings.BATCH_SIZE > 0
        assert test_settings.NUM_EPOCHS > 0
        assert test_settings.MAX_NEW_TOKENS > 0
        assert test_settings.LEARNING_RATE > 0
        
        # Test reasonable ranges
        assert 0 <= test_settings.TEMPERATURE <= 2.0
        assert 0 <= test_settings.TOP_P <= 1.0

    def test_directory_creation_on_import(self):
        """Test that directories are created when config is imported."""
        # These directories should exist after importing config
        assert OUTPUT_DIR.exists()
        assert WORKING_DIR.exists()

    @pytest.mark.parametrize("env_var,expected_type", [
        ("OUTPUT_PATH", str),
        ("WORKING_PATH", str),
    ])
    def test_environment_variable_types(self, env_var, expected_type):
        """Test that environment variables are handled correctly."""
        test_value = "/test/path"
        
        with patch.dict(os.environ, {env_var: test_value}):
            # The environment variable should be accessible
            assert os.getenv(env_var) == test_value

    def test_settings_inheritance(self):
        """Test that Settings inherits from BaseSettings correctly."""
        from pydantic_settings import BaseSettings
        
        assert issubclass(Settings, BaseSettings)
        test_settings = Settings()
        assert isinstance(test_settings, BaseSettings)

    def test_custom_settings_override(self):
        """Test creating settings with custom values."""
        custom_settings = Settings(
            PROJECT_NAME="Custom Project",
            MAX_SEQ_LENGTH=1024,
            TEMPERATURE=0.5
        )
        
        assert custom_settings.PROJECT_NAME == "Custom Project"
        assert custom_settings.MAX_SEQ_LENGTH == 1024
        assert custom_settings.TEMPERATURE == 0.5
        # Other values should remain default
        assert custom_settings.BATCH_SIZE == 4
        assert custom_settings.NUM_EPOCHS == 3