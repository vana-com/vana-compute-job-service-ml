from fastapi import APIRouter
from app.models.health import HealthCheckResponse
from app.services import HealthService

router = APIRouter()
health_service = HealthService()

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
    return health_service.check_system_health()
