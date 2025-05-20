# Vana Inference Engine

A FastAPI-based inference/training engine for Vana using Poetry and Unsloth.

## Overview

This project implements a long-running compute service for training and inference of language models. It provides two main endpoints:

1. `/train` - For training and fine-tuning language models using query results
2. `/chat/completions` - OpenAI-compatible API for chat completions

The application is designed to run as a long-running service inside a TEE (Trusted Execution Environment) and communicate with the Compute Engine via HTTP.

## Features

- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions API
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
│   ├── inference.py    # Inference API endpoints (OpenAI-compatible)
│   └── training.py     # Training API endpoints
└── utils/              # Utility functions
    ├── db.py           # Database operations
    └── events.py       # Event handling for SSE
scripts/
  |── image-build.sh    # Build the Docker image
  |── image-run.sh      # Run the Docker image
  |── image-export.sh   # Export the Docker image to `.tar` file. (gzip + sha256 manually)
```

## Data Flow

1. The application receives a training request with query parameters or a query ID
2. The application submits queries to the Query Engine.
  a. An existing query id can be supplied instead for query result reuse.
3. Input data is downloaded from the Query Engine API to the `WORKING_PATH` directory as `<query_id>.db`.
4. The application processes this data for training or uses it for inference.
5. Output models and artifacts are saved to the `OUTPUT_PATH` directory.
  a. Artifacts are scanned periodically and provided via the Compute Engine API for download. Created artifacts can be listed through the job status endpoint.
6. Working files and model caches are stored in the `WORKING_PATH` directory.

## Quick Start

1. Build the Docker image:
   ```bash
   ./scripts/image-build.sh
   ```

2. Run the container:
   ```bash
   ./scripts/image-run.sh
   ```

3. Access the API at http://localhost:8000

4. Export the Docker image for deployment:
   ```bash
   ./scripts/image-export.sh
   ```

## API Endpoints

### OpenAI-Compatible Chat Completions

- `POST /chat/completions`: Generate chat completions (OpenAI-compatible)
  ```json
  {
    "model": "my_finetuned_model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": false
  }
  ```

- `GET /models`: List all available models (OpenAI-compatible)

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
      "compute_job_id": 12,
      "refiner_id": 12,
      "query": "SELECT * FROM tweets WHERE user_id = ? ORDER BY created_at DESC LIMIT 100",
      "query_signature": "<signed query contents>",
      "params": ["user123"]
    }
  }
  ```

- `GET /train/{training_job_id}`: Get the status of a training job

- `GET /train/{training_job_id}/events`: Stream training events in real-time using SSE

- `GET /train/{training_job_id}/events/history`: Get the history of training events

## OpenAI API Compatibility

The inference API is designed to be compatible with OpenAI's API, allowing you to use it as a drop-in replacement for applications built with the OpenAI SDK:

### Python Example

```python
import openai

# Configure the client to use your Vana Inference Engine
client = openai.OpenAI(
    base_url="http://localhost:8000",  # Your Vana Inference Engine URL
    api_key="dummy-key"  # Not used but required by the client
)

# Use the same API as you would with OpenAI
response = client.chat.completions.create(
    model="my_finetuned_model",  # Your fine-tuned model name
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is machine learning?"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)
```

### JavaScript Example

```javascript
import OpenAI from 'openai';

// Configure the client to use your Vana Inference Engine
const openai = new OpenAI({
  baseURL: 'http://localhost:8000',  // Your Vana Inference Engine URL
  apiKey: 'dummy-key'  // Not used but required by the client
});

// Use the same API as you would with OpenAI
async function main() {
  const response = await openai.chat.completions.create({
    model: 'my_finetuned_model',  // Your fine-tuned model name
    messages: [
      { role: 'system', content: 'You are a helpful assistant.' },
      { role: 'user', content: 'What is machine learning?' }
    ],
    temperature: 0.7,
    max_tokens: 512
  });

  console.log(response.choices[0].message.content);
}

main();
```

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
   - Connect to `/train/{training_job_id}/events` to receive real-time updates
   - Use `/train/{training_job_id}/events/history` to get all past events

## Environment Variables

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

## Model Setup and Management

### Model Placement for Docker Setup

When running the application using Docker, models are managed through three mounted directories:

1. **Input Directory** (`./input` → `/mnt/input`): Contains input data such as query results used for training.
2. **Output Directory** (`./output` → `/mnt/output`): Stores trained and fine-tuned models that can be used for inference.
3. **Working Directory** (`./working` → `/mnt/working`): Contains temporary files and model caches used during training.

To set up models for inference:

1. Place your pre-trained models in the `./output` directory on your host machine. Each model should be in its own subdirectory.
2. The directory structure should be:
   ```
   ./output/
   ├── model_name_1/
   │   ├── config.json
   │   ├── tokenizer.json
   │   ├── model.safetensors
   │   └── metadata.json (optional)
   ├── model_name_2/
   │   └── ...
   ```

3. When running the Docker container, these models will be accessible at `/mnt/output` inside the container.

### Setting Up Models for Inference

There are two ways to set up models for inference:

#### 1. Using the Training API

The easiest way to set up a model is to train it using the `/train` endpoint:

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "meta-llama/Llama-2-7b-hf",
    "output_dir": "my_custom_model",
    "training_params": {
      "num_epochs": 3,
      "learning_rate": 2e-4,
      "batch_size": 4
    },
    "query_params": {
      "query": "SELECT * FROM your_data_table",
      "params": []
    }
  }'
```

This will:
- Download the base model (if not already cached)
- Train it using the provided query data
- Save the fine-tuned model to `/mnt/output/my_custom_model`

#### 2. Manually Copying Pre-trained Models

You can also manually copy pre-trained models to the `./output` directory:

1. Create a subdirectory with your model name in the `./output` directory
2. Copy all model files (config.json, tokenizer.json, model.safetensors, etc.) to this directory
3. Optionally, create a `metadata.json` file with information about the model

### Using Models for Inference

Once a model is available in the output directory, you can use it for inference via the OpenAI-compatible API:

```bash
curl -X POST http://localhost:8000/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "my_custom_model",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is machine learning?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
  }'
```

### Listing Available Models

To see all available models:

```bash
curl http://localhost:8000/models
```

This will return a list of all models in the `/mnt/output` directory that have a valid `config.json` file.
