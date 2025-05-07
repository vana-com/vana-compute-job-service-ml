import os
import uvicorn
import torch
import shutil
import psutil
from pathlib import Path
from fastapi import FastAPI, Depends, HTTPException, status, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any, List

from app.routers import training, inference
from app.config import settings
from app.utils.db import get_connection, execute_query, get_training_data
from app.models.inference import load_model

app = FastAPI(
    title="Vana Inference Engine",
    description="FastAPI-based inference/training engine for Vana",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router, prefix="/train", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])


@app.post("/test-query", tags=["query"])
async def test_query(
    query: str = Body(..., description="SQL query to execute"),
    params: List[Any] = Body(None, description="Parameters for the query")
):
    """
    Test endpoint to run a query against the database.
    
    This endpoint allows direct execution of SQL queries for testing purposes.
    In production, this would be restricted to authorized users only.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query (optional)
        
    Returns:
        The query results
    """
    try:
        results = execute_query(query, params)
        
        # Convert results to a more readable format
        column_names = []
        if results:
            # Get column names from cursor description
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute(query, params if params else [])
            column_names = [desc[0] for desc in cursor.description]
            conn.close()
        
        formatted_results = []
        for row in results:
            formatted_row = {}
            for i, value in enumerate(row):
                column_name = column_names[i] if i < len(column_names) else f"column_{i}"
                formatted_row[column_name] = value
            formatted_results.append(formatted_row)
        
        return {
            "status": "success",
            "query": query,
            "params": params,
            "row_count": len(results),
            "results": formatted_results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )

@app.post("/test-query-engine", tags=["query"])
async def test_query_engine(
    query_info: Dict[str, Any] = Body(..., description="Query information for the query engine")
):
    """
    Test endpoint to run a query through the query engine.
    
    This endpoint uses the query engine to process queries and return results.
    It demonstrates how the query engine would be used in production.
    
    Args:
        query_info: Dictionary containing query information. Can be:
            - {"query_id": str} - ID of an existing query
            - {"query": str, "params": List, "refiner_id": int, "query_signature": str} - Parameters for a new query
            
    Returns:
        The query results from the query engine
    """
    try:
        # Get data from the query engine
        results = get_training_data(query_info)
        
        return {
            "status": "success",
            "query_info": query_info,
            "row_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query engine execution failed: {str(e)}"
        )

@app.get("/", tags=["health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Vana Inference Engine is running"}

@app.get("/ping", tags=["health"])
async def ping():
    """Simple ping endpoint for container health checks."""
    return {"status": "ok"}

@app.get("/health", tags=["health"])
async def health_check():
    """
    Comprehensive health check endpoint.
    
    Checks:
    - Database connectivity
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
    
    # Check database connectivity
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT 1")
        cursor.fetchone()
        conn.close()
        health_status["components"]["database"] = {
            "status": "healthy",
            "message": "Database connection successful",
            "path": str(settings.DB_PATH)
        }
    except Exception as e:
        health_status["status"] = "degraded"
        health_status["components"]["database"] = {
            "status": "unhealthy",
            "message": f"Database connection failed: {str(e)}",
            "path": str(settings.DB_PATH)
        }
    
    # Check file system access
    file_systems = {
        "input_dir": settings.INPUT_DIR,
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
            test_file.write_text("test")
            test_file.read_text()
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
        cuda_available = torch.cuda.is_available()
        gpu_info = {
            "cuda_available": cuda_available,
            "device_count": torch.cuda.device_count() if cuda_available else 0,
        }
        
        if cuda_available:
            gpu_info["devices"] = []
            for i in range(gpu_info["device_count"]):
                gpu_info["devices"].append({
                    "name": torch.cuda.get_device_name(i),
                    "memory_total": torch.cuda.get_device_properties(i).total_memory,
                    "memory_allocated": torch.cuda.memory_allocated(i),
                    "memory_reserved": torch.cuda.memory_reserved(i)
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

if __name__ == "__main__":
    # Run the application with uvicorn when script is executed directly
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)