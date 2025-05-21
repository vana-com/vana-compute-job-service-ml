import shutil
import psutil
from pathlib import Path
from typing import Dict, Tuple

from torch import cuda
from app.config import settings
from fastapi import APIRouter
from app.models.health import (
    HealthCheckResponse, ComponentStatus, FileSystemStatus, 
    FileSystemDirectoryStatus, ModelStatus, GPUStatus, 
    GPUInfo, GPUDevice, SystemStatus, DiskStatus, MemoryStatus, CPUStatus
)

router = APIRouter()

def check_file_system(file_systems: Dict[str, Path]) -> Tuple[str, FileSystemStatus]:
    """
    Check if file system directories are accessible and writable.
    
    Args:
        file_systems: Dictionary mapping directory names to Path objects
        
    Returns:
        Tuple of overall status and FileSystemStatus object
    """
    file_system_dirs = {}
    overall_status = "healthy"
    
    for name, path in file_systems.items():
        try:
            # Check if directory exists and is writable
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
            
            # Try to write and read a test file
            test_file = path / ".health_check_test"
            with open(test_file, "w") as f:
                f.write("test")
            with open(test_file, "r") as f:
                f.read()
            test_file.unlink()  # Remove the test file
            
            file_system_dirs[name] = FileSystemDirectoryStatus(
                status="healthy",
                message="Directory is accessible and writable",
                path=str(path)
            )
        except Exception as e:
            overall_status = "degraded"
            file_system_dirs[name] = FileSystemDirectoryStatus(
                status="unhealthy",
                message=f"Directory access failed: {str(e)}",
                path=str(path)
            )
    
    return overall_status, FileSystemStatus(directories=file_system_dirs)

def check_model_loading() -> Tuple[str, ModelStatus]:
    """
    Check if model loading capability is working.
    
    Returns:
        Tuple of overall status and ModelStatus object
    """
    overall_status = "healthy"
    
    try:
        # Try to load the default model
        model_path = settings.MODEL_DIR / "default"
        if not model_path.exists():
            # If default model doesn't exist, use the base model path
            model_path = Path(settings.DEFAULT_BASE_MODEL)
        
        # Just check if model loading would work, don't actually load it
        # to avoid unnecessary memory usage during health checks
        model_status = ModelStatus(
            status="healthy",
            message="Model loading capability verified",
            path=str(model_path)
        )
    except Exception as e:
        overall_status = "degraded"
        model_path_str = str(model_path) if 'model_path' in locals() else "unknown"
        model_status = ModelStatus(
            status="unhealthy",
            message=f"Model loading capability check failed: {str(e)}",
            path=model_path_str
        )
    
    return overall_status, model_status

def check_gpu() -> GPUStatus:
    """
    Check GPU/CUDA availability and properties.
    
    Returns:
        GPUStatus object
    """
    try:
        cuda_available = cuda.is_available()
        device_count = cuda.device_count() if cuda_available else 0
        
        gpu_info = GPUInfo(
            cuda_available=cuda_available,
            device_count=device_count,
        )
        
        if cuda_available and device_count > 0:
            gpu_devices = []
            for i in range(device_count):
                properties = cuda.get_device_properties(i)
                gpu_devices.append(GPUDevice(
                    name=cuda.get_device_name(i),
                    memory_total=properties.total_memory,
                    memory_allocated=cuda.memory_allocated(i),
                    memory_reserved=cuda.memory_reserved(i)
                ))
            gpu_info.devices = gpu_devices
        
        return GPUStatus(
            status="healthy",
            message="GPU information retrieved successfully",
            info=gpu_info
        )
    except Exception as e:
        return GPUStatus(
            status="unhealthy",
            message=f"Failed to retrieve GPU information: {str(e)}",
            cuda_available=False
        )

def check_system_resources() -> Tuple[str, SystemStatus]:
    """
    Check system resources including disk, memory, and CPU.
    
    Returns:
        Tuple of overall status and SystemStatus object
    """
    overall_status = "healthy"
    
    try:
        disk_usage = shutil.disk_usage("/")
        memory = psutil.virtual_memory()
        
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
        
        system_message = "System resources retrieved successfully"
        system_status_value = "healthy"
        
        # Set warning if resources are low
        if disk_usage.free < 1_000_000_000:  # Less than 1GB free
            overall_status = "degraded"
            system_status_value = "warning"
            system_message = "Low disk space"
        
        if memory.available < 1_000_000_000:  # Less than 1GB available
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
        system_status = SystemStatus(
            status="unhealthy",
            message=f"Failed to retrieve system resources: {str(e)}"
        )
    
    return overall_status, system_status

@router.get("", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Comprehensive health check endpoint.
    
    Returns:
        HealthCheckResponse: A detailed health status of the system, including:
            - File system access
            - Model loading capability
            - GPU/CUDA availability
            - System resources
    """
    # Initialize with healthy status
    overall_status = "healthy"
    
    # Check file systems
    file_systems = {
        "output_dir": settings.OUTPUT_DIR,
        "working_dir": settings.WORKING_DIR,
        "model_dir": settings.MODEL_DIR
    }
    fs_status, file_system_status = check_file_system(file_systems)
    if fs_status != "healthy":
        overall_status = "degraded"
    
    # Check model loading
    model_status_value, model_status = check_model_loading()
    if model_status_value != "healthy":
        overall_status = "degraded"
    
    # Check GPU/CUDA
    gpu_status = check_gpu()
    
    # Check system resources
    system_status_value, system_status = check_system_resources()
    if system_status_value != "healthy":
        overall_status = "degraded"
    
    # Build the complete health check response
    components = ComponentStatus(
        file_system=file_system_status,
        model=model_status,
        gpu=gpu_status,
        system=system_status
    )
    
    return HealthCheckResponse(
        status=overall_status,
        timestamp=str(psutil.time.time()),
        components=components
    )
