import psutil

from app.config import settings
from app.services.health import check_file_system, check_model_loading, check_gpu, check_system_resources
from fastapi import APIRouter
from app.models.health import HealthCheckResponse, ComponentStatus

router = APIRouter()

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
