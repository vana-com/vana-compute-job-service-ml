import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import training, inference, health

app = FastAPI(
    title="Vana Inference Engine",
    description="FastAPI-based inference/training engine for Vana",
    version="0.1.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(training.router, prefix="/train", tags=["training"])
app.include_router(inference.router, prefix="/inference", tags=["inference"])
app.include_router(health.router, prefix="/health", tags=["health"])

if __name__ == "__main__":
    # Run the application with uvicorn when script is executed directly
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)