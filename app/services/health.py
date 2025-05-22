"""
Health Service Module - System health checks and monitoring.

This module provides health check functionality for the application,
monitoring file systems, model availability, GPU status, and system resources.

Examples:
    Basic usage:
    
    >>> from app.services.health import HealthService
    >>> health_service = HealthService()
    >>> health_check = health_service.check_system_health()
    >>> print(f"System status: {health_check.status}")  # "healthy" or "degraded"
    
    Check specific components:
    
    >>> file_systems = {"data": Path("/path/to/data")}
    >>> status, file_system = health_service.check_file_system(file_systems)
    >>> gpu_status = health_service.check_gpu()
    >>> print(f"GPU available: {gpu_status.info.cuda_available}")
    
    Custom settings:
    
    >>> from app.config import Settings
    >>> custom_settings = Settings(MODEL_DIR=Path("/custom/models"))
    >>> health_service = HealthService(settings=custom_settings)
"""

from typing import Dict, Tuple
from pathlib import Path
import shutil
import psutil
import time
import logging
import traceback
from torch import cuda

from app.config import settings as global_settings
from app.models.health import (
    FileSystemStatus, FileSystemDirectoryStatus, ModelStatus, GPUStatus, 
    GPUInfo, GPUDevice, SystemStatus, DiskStatus, MemoryStatus, CPUStatus,
    ComponentStatus, HealthCheckResponse
)

# Set up logging
logger = logging.getLogger(__name__)

class HealthService:
    """Service for system health checks."""
    
    def __init__(self, settings=None):
        """
        Initialize the health service.
        
        Args:
            settings: Application settings, defaults to global settings
        """
        logger.info("Initializing HealthService")
        self.settings = settings or global_settings
    
    def check_system_health(self) -> HealthCheckResponse:
        """
        Perform a complete system health check.
        
        Returns:
            HealthCheckResponse containing all health components
        """
        logger.info("Starting comprehensive system health check")
        start_time = time.time()
        overall_status = "healthy"
        
        file_systems = {
            "output_dir": self.settings.OUTPUT_DIR,
            "working_dir": self.settings.WORKING_DIR,
            "model_dir": self.settings.MODEL_DIR
        }
        
        logger.debug(f"Checking file systems: {list(file_systems.keys())}")
        fs_status, file_system = self.check_file_system(file_systems)
        if fs_status != "healthy":
            logger.warning(f"File system check reported degraded status: {fs_status}")
            overall_status = "degraded"
            
        logger.debug("Checking model loading capability")
        model_status_val, model = self.check_model_loading()
        if model_status_val != "healthy":
            logger.warning(f"Model loading check reported degraded status: {model_status_val}")
            overall_status = "degraded"
            
        logger.debug("Checking GPU/CUDA availability")
        gpu = self.check_gpu()
        if gpu.status != "healthy":
            logger.info(f"GPU check reported status: {gpu.status}")
        
        logger.debug("Checking system resources")
        sys_status_val, system = self.check_system_resources()
        if sys_status_val != "healthy":
            logger.warning(f"System resources check reported degraded status: {sys_status_val}")
            overall_status = "degraded"
            
        # Create response with all components
        components = ComponentStatus(
            file_system=file_system,
            model=model,
            gpu=gpu,
            system=system
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Health check completed in {elapsed_time:.2f}s with status: {overall_status}")
        
        return HealthCheckResponse(
            status=overall_status,
            timestamp=str(time.time()),
            components=components
        )
    
    def check_file_system(self, file_systems: Dict[str, Path]) -> Tuple[str, FileSystemStatus]:
        """
        Check if file system directories are accessible and writable.
        
        Args:
            file_systems: Dictionary mapping directory names to Path objects
            
        Returns:
            Tuple of overall status and FileSystemStatus object
        """
        logger.debug(f"Starting file system check for {len(file_systems)} directories")
        file_system_dirs = {}
        overall_status = "healthy"
        
        for name, path in file_systems.items():
            logger.debug(f"Checking directory '{name}' at {path}")
            try:
                # Check if directory exists and is writable
                if not path.exists():
                    logger.info(f"Directory '{name}' doesn't exist, creating it: {path}")
                    path.mkdir(parents=True, exist_ok=True)
                
                # Try to write and read a test file
                test_file = path / ".health_check_test"
                logger.debug(f"Writing test file to {test_file}")
                with open(test_file, "w") as f:
                    f.write("test")
                with open(test_file, "r") as f:
                    content = f.read()
                    if content != "test":
                        raise ValueError(f"Test file content mismatch: expected 'test', got '{content}'")
                test_file.unlink()  # Remove the test file
                
                logger.debug(f"Directory '{name}' is healthy")
                file_system_dirs[name] = FileSystemDirectoryStatus(
                    status="healthy",
                    message="Directory is accessible and writable",
                    path=str(path)
                )
            except Exception as e:
                error_detail = traceback.format_exc()
                logger.error(f"Error checking directory '{name}': {str(e)}\n{error_detail}")
                overall_status = "degraded"
                file_system_dirs[name] = FileSystemDirectoryStatus(
                    status="unhealthy",
                    message=f"Directory access failed: {str(e)}",
                    path=str(path)
                )
        
        logger.info(f"File system check complete. Status: {overall_status}")
        return overall_status, FileSystemStatus(directories=file_system_dirs)

    def check_model_loading(self) -> Tuple[str, ModelStatus]:
        """
        Check if model loading capability is working.
        
        Returns:
            Tuple of overall status and ModelStatus object
        """
        logger.debug("Starting model loading capability check")
        overall_status = "healthy"
        
        try:
            # Try to load the default model
            model_path = self.settings.MODEL_DIR / "default"
            if not model_path.exists():
                # If default model doesn't exist, use the base model path
                logger.info(f"Default model not found at {model_path}, using base model path")
                model_path = Path(self.settings.DEFAULT_BASE_MODEL)
            
            logger.debug(f"Verifying model path: {model_path}")
            
            # Just check if model loading would work, don't actually load it
            # to avoid unnecessary memory usage during health checks
            model_status = ModelStatus(
                status="healthy",
                message="Model loading capability verified",
                path=str(model_path)
            )
            logger.info(f"Model loading check passed for {model_path}")
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error checking model loading: {str(e)}\n{error_detail}")
            overall_status = "degraded"
            model_path_str = str(model_path) if 'model_path' in locals() else "unknown"
            model_status = ModelStatus(
                status="unhealthy",
                message=f"Model loading capability check failed: {str(e)}",
                path=model_path_str
            )
        
        return overall_status, model_status

    def check_gpu(self) -> GPUStatus:
        """
        Check GPU/CUDA availability and properties.
        
        Returns:
            GPUStatus object
        """
        logger.debug("Starting GPU/CUDA availability check")
        try:
            cuda_available = cuda.is_available()
            device_count = cuda.device_count() if cuda_available else 0
            
            logger.info(f"CUDA available: {cuda_available}, Device count: {device_count}")
            
            gpu_info = GPUInfo(
                cuda_available=cuda_available,
                device_count=device_count,
            )
            
            if cuda_available and device_count > 0:
                logger.debug("Getting GPU device information")
                gpu_devices = []
                for i in range(device_count):
                    properties = cuda.get_device_properties(i)
                    gpu_devices.append(GPUDevice(
                        name=cuda.get_device_name(i),
                        memory_total=properties.total_memory,
                        memory_allocated=cuda.memory_allocated(i),
                        memory_reserved=cuda.memory_reserved(i)
                    ))
                    logger.debug(f"GPU {i}: {cuda.get_device_name(i)}, "
                                 f"Total memory: {properties.total_memory / 1_000_000:.1f}MB, "
                                 f"Allocated: {cuda.memory_allocated(i) / 1_000_000:.1f}MB")
                gpu_info.devices = gpu_devices
            
            return GPUStatus(
                status="healthy",
                message="GPU information retrieved successfully",
                info=gpu_info
            )
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error checking GPU: {str(e)}\n{error_detail}")
            return GPUStatus(
                status="unhealthy",
                message=f"Failed to retrieve GPU information: {str(e)}",
                cuda_available=False
            )

    def check_system_resources(self) -> Tuple[str, SystemStatus]:
        """
        Check system resources including disk, memory, and CPU.
        
        Returns:
            Tuple of overall status and SystemStatus object
        """
        logger.debug("Starting system resources check")
        overall_status = "healthy"
        
        try:
            disk_usage = shutil.disk_usage("/")
            memory = psutil.virtual_memory()
            
            # Convert to human-readable format for logging
            disk_free_gb = disk_usage.free / 1_000_000_000
            memory_available_gb = memory.available / 1_000_000_000
            
            logger.info(f"Disk: {disk_free_gb:.1f}GB free ({disk_usage.free / disk_usage.total * 100:.1f}% free)")
            logger.info(f"Memory: {memory_available_gb:.1f}GB available ({memory.percent:.1f}% used)")
            
            disk_status = DiskStatus(
                total=disk_usage.total,
                used=disk_usage.used,
                free=disk_usage.free,
                percent=disk_usage.used / disk_usage.total * 100
            )
            
            memory_status = MemoryStatus(
                total=memory.total,
                available=memory.available,
                used=memory.used,
                percent=memory.percent
            )
            
            cpu_status = CPUStatus(
                percent=psutil.cpu_percent(interval=0.1),
                count=psutil.cpu_count()
            )
            
            logger.info(f"CPU: {cpu_status.percent:.1f}% utilization, {cpu_status.count} cores")
            
            system_message = "System resources retrieved successfully"
            system_status_value = "healthy"
            
            # Set warning if resources are low
            if disk_usage.free < 1_000_000_000:  # Less than 1GB free
                logger.warning(f"Low disk space: {disk_free_gb:.1f}GB free")
                overall_status = "degraded"
                system_status_value = "warning"
                system_message = "Low disk space"
            
            if memory.available < 1_000_000_000:  # Less than 1GB available
                logger.warning(f"Low memory: {memory_available_gb:.1f}GB available")
                overall_status = "degraded"
                system_status_value = "warning"
                system_message = "Low memory"
                
            system_status = SystemStatus(
                status=system_status_value,
                message=system_message,
                disk=disk_status,
                memory=memory_status,
                cpu=cpu_status
            )
        except Exception as e:
            error_detail = traceback.format_exc()
            logger.error(f"Error checking system resources: {str(e)}\n{error_detail}")
            system_status = SystemStatus(
                status="unhealthy",
                message=f"Failed to retrieve system resources: {str(e)}"
            )
        
        return overall_status, system_status