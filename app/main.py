import os
import uvicorn
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.routers import training, inference
from app.config import settings

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

@app.get("/", tags=["health"])
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Vana Inference Engine is running"}

@app.get("/ping", tags=["health"])
async def ping():
    """Simple ping endpoint for container health checks."""
    return {"status": "ok"}

if __name__ == "__main__":
    # Run the application with uvicorn when script is executed directly
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)