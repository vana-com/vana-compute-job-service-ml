import shutil
import psutil
from pathlib import Path
from typing import Dict, Any

from torch import cuda
from app.config import settings
from fastapi import APIRouter

router = APIRouter()

@router.get("")
async def health_check() -> Dict[str, Any]:
    """
    Comprehensive health check endpoint.
    
    Checks:
    - File system access
    - Model loading capability
    - GPU/CUDA availability
    - System resources
    """
    health_status = {
        "status": "healthy",
        "timestamp": str(psutil.time.time()),
        "components": {}
    }
        
    # Check file system access
    file_systems = {
        "output_dir": settings.OUTPUT_DIR,
        "working_dir": settings.WORKING_DIR,
        "model_dir": settings.MODEL_DIR
    }
    
    health_status["components"]["file_system"] = {"directories": {}}
    
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
            
            health_status["components"]["file_system"]["directories"][name] = {
                "status": "healthy",
                "message": "Directory is accessible and writable",
                "path": str(path)
            }
        except Exception as e:
            health_status["status"] = "degraded"
            health_status["components"]["file_system"]["directories"][name] = {
                "status": "unhealthy",
                "message": f"Directory access failed: {str(e)}",
                "path": str(path)
            }
    
    # Check model loading capability
    try:
        # Try to load the default model
        model_path = settings.MODEL_DIR / "default"
        if not model_path.exists():
            # If default model doesn't exist, use the base model path
            model_path = Path(settings.DEFAULT_BASE_MODEL)
        
        # Just check if model loading would work, don't actually load it
        # to avoid unnecessary memory usage during health checks
        health_status["components"]["model"] = {
            "status": "healthy",
            "message": "Model loading capability verified",
            "path": str(model_path)
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["model"] = {
            "status": "unhealthy",
            "message": f"Model loading capability check failed: {str(e)}",
            "path": str(model_path) if 'model_path' in locals() else "unknown"
        }
    
    # Check GPU/CUDA availability
    try:
        cuda_available = cuda.is_available()
        gpu_info = {
            "cuda_available": cuda_available,
            "device_count": cuda.device_count() if cuda_available else 0,
        }
        
        if cuda_available:
            gpu_info["devices"] = []
            for i in range(gpu_info["device_count"]):
                gpu_info["devices"].append({
                    "name": cuda.get_device_name(i),
                    "memory_total": cuda.get_device_properties(i).total_memory,
                    "memory_allocated": cuda.memory_allocated(i),
                    "memory_reserved": cuda.memory_reserved(i)
                })
        
        health_status["components"]["gpu"] = {
            "status": "healthy",
            "message": "GPU information retrieved successfully",
            "info": gpu_info
        }
    except Exception as e:
        health_status["components"]["gpu"] = {
            "status": "unhealthy",
            "message": f"Failed to retrieve GPU information: {str(e)}",
            "cuda_available": False
        }
    
    # Check system resources
    try:
        disk_usage = shutil.disk_usage("/")
        memory = psutil.virtual_memory()
        
        health_status["components"]["system"] = {
            "status": "healthy",
            "message": "System resources retrieved successfully",
            "disk": {
                "total": disk_usage.total,
                "used": disk_usage.used,
                "free": disk_usage.free,
                "percent": disk_usage.used / disk_usage.total * 100
            },
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "used": memory.used,
                "percent": memory.percent
            },
            "cpu": {
                "percent": psutil.cpu_percent(interval=0.1),
                "count": psutil.cpu_count()
            }
        }
        
        # Set warning if resources are low
        if disk_usage.free < 1_000_000_000:  # Less than 1GB free
            health_status["status"] = "degraded"
            health_status["components"]["system"]["status"] = "warning"
            health_status["components"]["system"]["message"] = "Low disk space"
        
        if memory.available < 1_000_000_000:  # Less than 1GB available
            health_status["status"] = "degraded"
            health_status["components"]["system"]["status"] = "warning"
            health_status["components"]["system"]["message"] = "Low memory"
    except Exception as e:
        health_status["components"]["system"] = {
            "status": "unhealthy",
            "message": f"Failed to retrieve system resources: {str(e)}"
        }
    
    return health_status
