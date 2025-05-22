from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

class ModelPermission(BaseModel):
    """Permission information for a model."""
    id: str
    object: Literal["model_permission"] = "model_permission"
    created: int
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "vana"
    group: Optional[str] = None
    is_blocking: bool = False

class ModelData(BaseModel):
    """Information about a specific model."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "vana"
    permission: List[ModelPermission] = []
    root: str
    parent: Optional[str] = None

class ModelListResponse(BaseModel):
    """Response model for listing available models."""
    object: Literal["list"] = "list"
    data: List[ModelData]
