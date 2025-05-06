# Vana Inference Engine

A FastAPI-based inference/training engine for Vana using Poetry and Unsloth.

## Overview

This project implements a long-running compute service for training and inference of language models. It provides two main endpoints:

1. `/train` - For training and fine-tuning language models using query results
2. `/inference` - For generating text using fine-tuned models

The application is designed to run as a long-running service inside a TEE (Trusted Execution Environment) and communicate with the Compute Engine via HTTP.

## Features

- **Query-Based Training**: Train models using the results of specific queries
- **Real-time Training Progress**: Monitor training progress in real-time using Server-Sent Events (SSE)
- **FastAPI Backend**: Provides a robust API for training and inference
- **Unsloth Integration**: Efficient fine-tuning of language models
- **Poetry Dependency Management**: Clean and reproducible dependency management
- **Background Training Jobs**: Long-running training jobs that don't block the API
- **Streaming Inference**: Support for streaming text generation responses
- **Model Management**: List and manage fine-tuned models

## Directory Structure

```
app/                    # Application code
├── config.py           # Application configuration
├── main.py             # FastAPI application entry point
├── models/             # ML model implementations
│   ├── inference.py    # Text generation using fine-tuned models
│   └── trainer.py      # Model training using Unsloth
├── routers/            # API route definitions
│   ├── inference.py    # Inference API endpoints
│   └── training.py     # Training API endpoints
└── utils/              # Utility functions
    ├── db.py           # Database operations
    └── events.py       # Event handling for SSE
```

## Data Flow

1. The application receives a training request with query parameters or a query ID
2. If query parameters are provided, the application would submit the query to the Query Engine (simulated in this implementation)
3. Input data is provided from the compute engine through a mounted `/mnt/input` directory containing query results
4. The application processes this data for training or uses it for inference
5. Output models and artifacts are saved to the `/mnt/output` directory
6. Working files and model caches are stored in the `/mnt/working` directory

## Quick Start

1. Build the Docker image:
   ```bash
   ./image-build.sh
   ```

2. Run the container:
   ```bash
   ./image-run.sh
   ```

3. Access the API at http://localhost:8000

4. Export the Docker image for deployment:
   ```bash
   ./image-export.sh
   ```

## API Endpoints

### Training

- `POST /train`: Start a training job using query results
  ```json
  {
    "model_name": "meta-llama/Llama-2-7b-hf",
    "output_dir": "my_finetuned_model",
    "training_params": {
      "num_epochs": 3,
      "learning_rate": 2e-4,
      "batch_size": 4
    },
    "query_params": {
      "query": "SELECT * FROM tweets WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
      "params": ["user123"],
      "refiner_id": 12
    }
  }
  ```

- `GET /train/{job_id}`: Get the status of a training job

- `GET /train/{job_id}/events`: Stream training events in real-time using SSE

- `GET /train/{job_id}/events/history`: Get the history of training events

### Inference

- `POST /inference`: Generate text using a fine-tuned model
  ```json
  {
    "model_path": "my_finetuned_model",
    "prompt": "What is machine learning?",
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": false
  }
  ```

- `GET /inference/models`: List all available fine-tuned models

## Query-Based Training

The training process uses the results of queries to the Query Engine:

1. **Query Specification**: You can specify a query in two ways:
   - Provide a `query_id` for an existing query
   - Provide `query_params` to execute a new query

2. **Query Parameters**:
   - `query`: The SQL query to execute
   - `params`: Parameters for the query
   - `refiner_id`: ID of the data refiner
   - `query_signature`: Signature for query authentication

3. **Data Processing**:
   - The query results are stored in the input database
   - The training process extracts and formats this data for fine-tuning
   - The model is trained on the formatted examples

## Real-time Training Progress

The application provides real-time updates on training progress using Server-Sent Events (SSE):

1. **Event Types**:
   - `init`: Initial event with job information
   - `status`: Status updates during different phases of training
   - `progress`: Regular progress updates during training
   - `log`: Log messages from the training process
   - `complete`: Sent when training is complete
   - `error`: Sent if an error occurs during training

2. **Monitoring Progress**:
   - Connect to `/train/{job_id}/events` to receive real-time updates
   - Use `/train/{job_id}/events/history` to get all past events

3. **Client Example**:
   ```javascript
   const eventSource = new EventSource('/train/train_1234abcd/events');
   
   eventSource.addEventListener('progress', (event) => {
     const data = JSON.parse(event.data);
     console.log(`Progress: ${data.progress}%, Step: ${data.step}/${data.total_steps}`);
     updateProgressBar(data.progress);
   });
   
   eventSource.addEventListener('complete', (event) => {
     console.log('Training completed!');
     eventSource.close();
   });
   
   eventSource.addEventListener('error', (event) => {
     console.error('Training error:', event.data);
     eventSource.close();
   });
   ```

## Environment Variables

- `INPUT_PATH`: Path to the input directory (default: `/mnt/input`)
- `OUTPUT_PATH`: Path to the output directory (default: `/mnt/output`)
- `WORKING_PATH`: Path to the working directory (default: `/mnt/working`)
- `PORT`: Port to run the FastAPI server on (default: `8000`)

## Deployment

For deployment in a TEE environment, follow these steps:

1. Build and export the Docker image:
   ```bash
   ./image-build.sh
   ./image-export.sh
   ```

2. Compress the image:
   ```bash
   gzip vana-inference.tar
   ```

3. Calculate the SHA256 checksum:
   ```bash
   sha256sum vana-inference.tar.gz | cut -d' ' -f1
   ```

4. Upload the image to a publicly accessible URL

5. Register the Compute Instruction on-chain with the image URL and SHA256 checksum

6. Get approval from the DLP owner

7. Register and submit the Compute Job for execution
