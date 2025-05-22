from pydantic import BaseModel
from typing import Dict, List, Optional, Literal

class FileSystemDirectoryStatus(BaseModel):
    """Status of a specific directory in the file system."""
    status: Literal["healthy", "unhealthy", "warning"]
    message: str
    path: str

class FileSystemStatus(BaseModel):
    """Status of the file system component."""
    directories: Dict[str, FileSystemDirectoryStatus]

class ModelStatus(BaseModel):
    """Status of the model loading capability."""
    status: Literal["healthy", "unhealthy", "warning"]
    message: str
    path: str

class GPUDevice(BaseModel):
    """Information about a specific GPU device."""
    name: str
    memory_total: int
    memory_allocated: int
    memory_reserved: int

class GPUInfo(BaseModel):
    """Information about GPU/CUDA availability."""
    cuda_available: bool
    device_count: int
    devices: Optional[List[GPUDevice]] = None

class GPUStatus(BaseModel):
    """Status of the GPU component."""
    status: Literal["healthy", "unhealthy", "warning"]
    message: str
    info: Optional[GPUInfo] = None
    cuda_available: Optional[bool] = None

class DiskStatus(BaseModel):
    """Status of the disk."""
    total: int
    used: int
    free: int
    percent: float

class MemoryStatus(BaseModel):
    """Status of the memory."""
    total: int
    available: int
    used: int
    percent: float

class CPUStatus(BaseModel):
    """Status of the CPU."""
    percent: float
    count: int

class SystemStatus(BaseModel):
    """Status of the system resources."""
    status: Literal["healthy", "unhealthy", "warning"]
    message: str
    disk: Optional[DiskStatus] = None
    memory: Optional[MemoryStatus] = None
    cpu: Optional[CPUStatus] = None

class ComponentStatus(BaseModel):
    """Status of all components."""
    file_system: FileSystemStatus
    model: ModelStatus
    gpu: GPUStatus
    system: SystemStatus

class HealthCheckResponse(BaseModel):
    """Response model for health check endpoint."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: str
    components: ComponentStatus
