# Vana Inference Engine

A FastAPI-based inference/training engine for Vana using Poetry and Unsloth.

## Overview

This application provides two main endpoints:

1. `/train` - For training and fine-tuning language models using query results
2. `/chat/completions` - OpenAI-compatible API for chat completions

The application is designed to run as a long-running service inside a TEE (Trusted Execution Environment) and communicate with the Compute Engine via HTTP.

## Architecture

- **FastAPI**: Provides the web server and API endpoints
- **Unsloth**: Efficient fine-tuning of language models
- **Poetry**: Dependency management
- **SQLite**: Access to query results from the input directory
- **Server-Sent Events (SSE)**: Real-time training progress updates
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions API

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

## OpenAI-Compatible API

The inference API is designed to be compatible with OpenAI's API, allowing you to use it as a drop-in replacement for applications built with the OpenAI SDK:

### Endpoints

- `POST /chat/completions`: Generate chat completions
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

- `GET /models`: List all available models

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

## API Endpoints

### Training

- `POST /train`: Start a training job
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

### Inference (OpenAI-Compatible)

- `POST /chat/completions`: Generate chat completions
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

- `GET /models`: List all available models

## Directory Structure

```
app/
├── config.py           # Application configuration
├── main.py             # FastAPI application entry point
├── models/             # ML model implementations
│   ├── inference.py    # Text generation using fine-tuned models (OpenAI-compatible)
│   └── trainer.py      # Model training using Unsloth
├── routers/            # API route definitions
│   ├── inference.py    # Inference API endpoints (OpenAI-compatible)
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

## Development

This application is built with Poetry for dependency management. To set up the development environment:

```bash
# Install dependencies
poetry install

# Run the application
poetry run python -m app.main
```

## Deployment

The application is containerized using Docker and can be deployed as a long-running service in a TEE.

```bash
# Build the Docker image
docker build -t vana-inference .

# Run the container
docker run -p 8000:8000 -v ./input:/mnt/input -v ./output:/mnt/output -v ./working:/mnt/working vana-inference
```

## Environment Variables

- `INPUT_PATH`: Path to the input directory (default: `/mnt/input`)
- `OUTPUT_PATH`: Path to the output directory (default: `/mnt/output`)
- `WORKING_PATH`: Path to the working directory (default: `/mnt/working`)
- `PORT`: Port to run the FastAPI server on (default: `8000`)