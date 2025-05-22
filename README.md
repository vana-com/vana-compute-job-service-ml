# Vana ML Service Compute Job Template (Training & Inference)

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20Community-7289DA?style=for-the-badge&logo=discord&logoColor=white)](https://t.co/VkqwlJZ4ph)
[![Documentation](https://img.shields.io/badge/Docs-Read%20More-0088CC?style=for-the-badge&logo=readthedocs&logoColor=white)](https://docs.vana.org/docs/home)
[![Website](https://img.shields.io/badge/Website-Visit%20Us-00ADEF?style=for-the-badge&logo=internetexplorer&logoColor=white)](https://www.vana.org/)
[![X (Twitter)](https://img.shields.io/badge/Twitter-Follow%20Us-000000?style=for-the-badge&logo=x&logoColor=white)](https://x.com/vana)

<details>
<summary>Table of Contents</summary>

- [Overview](#overview)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [System Prerequisites](#system-prerequisites)
- [Local Development](#local-development)
  - [Using Docker (Recommended)](#using-docker-recommended)
  - [Using FastAPI Directly (Limited Support)](#using-fastapi-directly-limited-support)
- [Using the Service Job](#using-the-service-job)
  - [Submitting a Long-Running Service Job](#1-submitting-a-long-running-service-job)
  - [Querying Job Status and Artifacts](#2-querying-job-status-and-artifacts)
  - [Interacting with the Service Job API](#3-interacting-with-the-service-job-api)
  - [Debugging Jobs](#4-debugging-jobs)
  - [Cancelling Jobs](#5-cancelling-jobs)
- [Smart Contracts](#smart-contracts)
- [Customizing Query Result Processing](#customizing-query-result-processing)
- [Complete Examples](#complete-examples)
  - [Training Workflow Example](#training-workflow-example)
  - [Inference Workflow Example](#inference-workflow-example)
- [API Reference](#api-reference)
  - [Service Job Endpoints](#service-job-endpoints)
  - [Compute Engine Endpoints](#compute-engine-endpoints)
- [Deployment](#deployment)

</details>
<br/>
A FastAPI-based service for running long-lived training and inference jobs within the Vana Compute Engine.

## Overview

This template implements a long-running compute service for training and inference of language models. It's designed to run inside the Vana Compute Engine as a containerized service job, exposing endpoints through the compute engine's proxy.

Key features:
- **OpenAI-Compatible API**: Drop-in replacement for OpenAI's chat completions API
- **Query-Based Training**: Train models using the results of database queries against the Vana Query Engine
- **Real-time Training Progress**: Monitor training progress via Server-Sent Events
- **Permissioned Infrastructure**: Works inside the Vana Compute Engine secure TEE infrastructure
- **Efficient Training**: Uses Unsloth for optimized fine-tuning of language models

## Project Structure

```
├── Dockerfile                      # Container definition for the service
├── README.md                       # Documentation
├── app/                            # Main application code
│   ├── config.py                   # Configuration and environment settings
│   ├── main.py                     # FastAPI application entry point
│   ├── ml/                         # Machine learning implementation
│   │   ├── inference.py            # Model loading and text generation
│   │   └── trainer.py              # Model training and fine-tuning with Unsloth
│   ├── models/                     # Data models and schemas
│   │   ├── health.py               # System health schemas
│   │   ├── inference.py            # Inference request/response schemas
│   │   ├── openai.py               # OpenAI-compatible API schemas
│   │   └── training.py             # Training job schemas
│   ├── routers/                    # API routes and endpoints
│   │   ├── health.py               # System health check endpoints
│   │   ├── inference.py            # Inference API endpoints
│   │   └── training.py             # Training API endpoints
│   ├── services/                   # Business logic implementation
│   │   ├── health.py               # System health service logic
│   │   ├── inference.py            # Inference service logic
│   │   └── training.py             # Training service logic
│   └── utils/                      # Utility functions
│       ├── db.py                   # Database and query result processing
│       ├── devices.py              # GPU/hardware detection and setup
│       ├── events.py               # Server-Sent Events (SSE) handling
│       └── query_engine_client.py  # Client for Vana Query Engine
├── pyproject.toml                  # Python dependencies and project metadata
└── scripts/                        # Deployment and development scripts
    ├── image-build.sh              # Build Docker image with TEE compatibility
    ├── image-export.sh             # Export Docker image for deployment
    └── image-run.sh                # Run Docker image locally for testing
```

## How It Works

When deployed, this service:
1. Runs as a containerized job within the Vana Compute Engine
2. Provides training and inference capabilities through proxied endpoints
3. Stores artifacts (trained models, logs) that can be accessed through the Compute Engine REST API
4. Maintains its state for as long as the service job runs

## System Prerequisites

- **CUDA Support**: This job requires CUDA-compatible GPUs. It's optimized for:
  - CUDA 12.4 or compatible
  - Compatible with PyTorch 2.6.0+cu124
  - Tested on NVIDIA T4, H200 and similar GPU architectures
- **Docker**: Required for building and running the containerized service
- **Python** (optional): Version 3.10 or higher for local development without Docker
- **Poetry** (optional): For local development without Docker

If you need to run on different hardware, you'll need to modify:
- Dependencies in `pyproject.toml` to match your CUDA version
- Potentially update model loading code for different hardware constraints

## Local Development

There are two ways to run the service locally on compatible machines:

### Using Docker (Recommended)

The Docker approach is the same method used in production and is the most reliable way to test your service:

1. Build the Docker image:
   ```bash
   ./scripts/image-build.sh
   ```
   
   This builds the image with TEE (AMD) compatibility for running in secure environments, ensuring the same environment as the production Vana Compute Engine.

2. Run the container locally:
   ```bash
   ./scripts/image-run.sh
   ```
   
   This script mounts the necessary directories, allocates all GPUs, and exposes port 8000 to interact with the service.

3. Access the service at `http://localhost:8000`

### Using FastAPI Directly (Limited Support)

For quick iterations during development, you can run the service directly with FastAPI and uvicorn on compatible machines:

1. Set env variables (`OUTPUT_PATH`, `WORKING_PATH`, `PORT=8000`)

2. Install dependencies:
   ```bash
   poetry install
   ```

3. Run the FastAPI application:
   ```bash
   poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

> **Note**: This method has limited support and may not fully replicate the containerized environment. Some features related to filesystem access or GPU integration might behave differently. Always test the final version using Docker before deployment.

## Using the Service Job

### 1. Submitting a Long-Running Service Job

To submit a service job, you need to:

1. Register the job on-chain through the Compute Engine smart contract
2. Submit the job to the Compute Engine API with the appropriate parameters

**Example: Submitting a job to the Compute Engine**

```python
import requests
from eth_account import Account
from eth_account.messages import encode_defunct

# Configuration
JOB_ID = "4976"  # Job ID from the smart contract
COMPUTE_ENGINE_URL = "http://compute-engine-url"
PRIVATE_KEY = "your_private_key"  # Private key for signing

def sign_message(message, private_key):
    """Sign a message using the provided private key."""
    account = Account.from_key(private_key)
    message_hash = encode_defunct(text=message)
    signed_message = account.sign_message(message_hash)
    return signed_message.signature.hex()

# Set up headers with job signature for authentication
headers = {
    "X-Job-ID-Signature": sign_message(JOB_ID, PRIVATE_KEY),
    "Content-Type": "application/json"
}

# Job creation request
job_request = {
    "input": {
        "refiner_id": 1,
        "query": "SELECT * FROM your_table",
        "query_signature": sign_message("SELECT * FROM your_table", PRIVATE_KEY),
        "params": []
    }
}

# Submit the job
response = requests.post(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}",
    headers=headers,
    json=job_request
)

print(f"Job submission response: {response.json()}")
```

> **IMPORTANT**: The `X-Job-ID-Signature` header containing the relevant job id signed with the job owner's wallet is required for all Compute Engine endpoints. Only the job owner can access these endpoints, as the signature proves ownership of the job's associated wallet.

### 2. Querying Job Status and Artifacts

Once a job is running, you can check its status and retrieve any artifacts it produces:

```python
# Get job status
status_response = requests.get(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}",
    headers=headers
)

job_result = status_response.json()
print(f"Job status: {job_result['status']}")

# List artifacts
if 'artifacts' in job_result:
    for artifact in job_result['artifacts']:
        print(f"Artifact: {artifact['id']} - {artifact['name']}")
        
        # Download an artifact
        artifact_response = requests.get(
            f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/artifacts/{artifact['id']}",
            headers=headers
        )
        
        if artifact_response.status_code == 200:
            with open(artifact['name'], 'wb') as f:
                f.write(artifact_response.content)
            print(f"Downloaded artifact to {artifact['name']}")
```

> **Note**: Trained models are stored in `{WORKING_PATH}/models` within the container, and model artifacts are exported as zip files to `{OUTPUT_PATH}/<model_name>.zip`. The Compute Engine periodically scans for new artifacts and makes them available through the job status endpoint.

### 3. Interacting with the Service Job API

The service job exposes its own API endpoints that can be accessed through the Compute Engine's proxy:

#### Training a Model

```python
# Start a training job
train_payload = {
    "model_name": "meta-llama/Llama-2-7b-hf",  # Base model for fine-tuning
    "output_dir": "user_data_model",  # Your trained model name
    "training_params": {
        "num_epochs": 3,
        "learning_rate": 2e-4,
        "batch_size": 4
    },
    "query_params": {
        "compute_job_id": JOB_ID,
        "refiner_id": 1,
        "query": "SELECT * FROM your_training_data",
        "query_signature": sign_message("SELECT * FROM your_training_data", PRIVATE_KEY),
        "params": []
    }
}

train_response = requests.post(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/train/",
    headers=headers,
    json=train_payload
)

training_job = train_response.json()
training_job_id = training_job['job_id']
print(f"Training job started: {training_job_id}")
```

#### Monitoring Training Progress

You can monitor training progress in real-time using Server-Sent Events (SSE):

```python
import sseclient

# Connect to SSE endpoint for real-time training updates
sse_url = f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/train/{training_job_id}/events"
sse_response = requests.get(sse_url, headers=headers, stream=True)
client = sseclient.SSEClient(sse_response)

for event in client.events():
    print(f"Training update: {event.data}")
```

#### Using the Trained Model for Inference

```python
# Generate text using a trained model
inference_payload = {
    "model": "my_finetuned_model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What insights can you provide from my data?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512
}

inference_response = requests.post(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/inference/chat/completions",
    headers=headers,
    json=inference_payload
)

result = inference_response.json()
print(f"Generated text: {result['choices'][0]['message']['content']}")
```

### 4. Debugging Jobs

Several tools are available for debugging your service jobs:

#### Streaming Container Logs

```python
# Stream logs from the container
logs_response = requests.get(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/logs",
    headers=headers,
    stream=True
)

for line in logs_response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

#### Using OpenAPI Documentation

Both the Compute Engine and your service job expose OpenAPI documentation:

- Compute Engine API docs: `{COMPUTE_ENGINE_URL}/docs` or `{COMPUTE_ENGINE_URL}/redoc`
- Service Job API docs: `{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/docs` or `{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/redoc`
- Raw OpenAPI schema: `{COMPUTE_ENGINE_URL}/openapi.json` or `{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/openapi.json`

These interactive documentation pages allow you to explore all available endpoints.

#### Training Event History

For training jobs, you can retrieve the history of all training events:

```python
# Get training event history
events_response = requests.get(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/train/{training_job_id}/events/history",
    headers=headers
)

events = events_response.json()
for event in events:
    print(f"Event type: {event['type']}, data: {event['data']}")
```

### 5. Cancelling Jobs

To cancel a running job, you can call the `cancelJob` method on the Compute Engine smart contract:

```javascript
// Solidity contract interaction example (web3.js)
const computeEngineContract = new web3.eth.Contract(
    ComputeEngineABI,
    COMPUTE_ENGINE_ADDRESS
);

// Cancel the job
await computeEngineContract.methods.cancelJob(JOB_ID).send({
    from: userAddress,
    gas: 200000
});
```

For Python applications, you can use web3.py:

```python
from web3 import Web3

# Connect to the blockchain
w3 = Web3(Web3.HTTPProvider('https://rpc.vana.org/'))  # Mainnet
# or w3 = Web3(Web3.HTTPProvider('https://rpc.moksha.vana.org/'))  # Testnet

# Load the contract
contract = w3.eth.contract(address=COMPUTE_ENGINE_ADDRESS, abi=COMPUTE_ENGINE_ABI)

# Cancel the job
tx_hash = contract.functions.cancelJob(int(JOB_ID)).transact({
    'from': w3.eth.accounts[0]
})

# Wait for the transaction to be mined
receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
print(f"Job cancelled: {receipt}")
```

## Smart Contracts

The following smart contracts are relevant for working with long-running compute jobs:

| Contract | Description | Mainnet Address | Testnet Address |
|----------|-------------|----------------|----------------|
| ComputeEngine | Main contract for job management | [0xb2BF...47bd](https://vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy) | [0xb2BF...47bd](https://moksha.vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy) |
| ComputeInstructionRegistry | Registry of compute instruction images | [0x5786...63A5](https://vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy) | [0x5786...63A5](https://moksha.vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy) |
| QueryEngine | Main contract for query permissioning | [0xd25E...0490](https://vanascan.io/address/0xd25Eb66EA2452cf3238A2eC6C1FD1B7F5B320490?tab=read_write_proxy) | [0xd25E...0490](https://moksha.vanascan.io/address/0xd25Eb66EA2452cf3238A2eC6C1FD1B7F5B320490?tab=read_write_proxy) |

## Customizing Query Result Processing

You can customize how query results are converted into training data by modifying the data processing pipeline in your service code.

The primary location to customize is in `utils/db.py`, where the `format_training_examples` method transforms raw database records into training pairs. Modify this function to match your specific data schema and training needs.

## Complete Examples

### Training Workflow Example

This example demonstrates a complete workflow for starting and monitoring a training job:

```python
import os
import requests
import json
import time
from eth_account import Account
from eth_account.messages import encode_defunct

# Configuration
JOB_ID = "4976"
COMPUTE_ENGINE_URL = "http://compute-engine-url"
PRIVATE_KEY = os.getenv("WALLET_PRIVATE_KEY")

def sign_message(message, private_key):
    account = Account.from_key(private_key)
    message_hash = encode_defunct(text=message)
    signed_message = account.sign_message(message_hash)
    return signed_message.signature.hex()

# Start a training job
headers = {
    "X-Job-ID-Signature": sign_message(JOB_ID, PRIVATE_KEY),
    "Content-Type": "application/json"
}

train_payload = {
    "model_name": "meta-llama/Llama-2-7b-hf",  # Base model for fine-tuning
    "output_dir": "user_data_model",  # Your trained model name
    "training_params": {
        "num_epochs": 6,
        "learning_rate": 2e-4,
        "batch_size": 4
    },
    "query_params": {
        "compute_job_id": JOB_ID,
        "refiner_id": 1,
        "query": """
            SELECT 
                u.user_id,
                u.locale,
                u.name,
                u.email,
                a.source AS auth_source
            FROM users AS u
            LEFT JOIN auth_sources AS a ON u.user_id = a.user_id
        """,
        "query_signature": sign_message("SELECT u.user_id, u.locale...", PRIVATE_KEY),
        "params": []
    }
}

response = requests.post(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/train/",
    headers=headers,
    json=train_payload
)

if response.status_code == 202:
    result = response.json()
    training_job_id = result['job_id']
    print(f"Training job started: {training_job_id}")
    
    # Monitor training progress
    for _ in range(10):
        status_response = requests.get(
            f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/train/{training_job_id}",
            headers=headers
        )
        
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"Status: {status['status']}")
            
            if status['status'] in ['complete', 'error']:
                break
        
        time.sleep(30)
else:
    print(f"Failed to start training: {response.text}")
```

### Inference Workflow Example

This example shows how to use a trained model for inference:

```python
import requests
from eth_account import Account
from eth_account.messages import encode_defunct

# Configuration
JOB_ID = "4976"
COMPUTE_ENGINE_URL = "http://compute-engine-url"
PRIVATE_KEY = "your_private_key"

def sign_message(message, private_key):
    account = Account.from_key(private_key)
    message_hash = encode_defunct(text=message)
    signed_message = account.sign_message(message_hash)
    return signed_message.signature.hex()

# Set up headers
headers = {
    "X-Job-ID-Signature": sign_message(JOB_ID, PRIVATE_KEY),
    "Content-Type": "application/json"
}

# List available models
models_response = requests.get(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/inference/models",
    headers=headers
)

if models_response.status_code == 200:
    models = models_response.json()
    print("Available models:")
    for model in models.get('data', []):
        print(f"  - {model.get('id')}")

# Generate text
inference_payload = {
    "model": "user_data_model",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the locale of user 4?"}
    ],
    "temperature": 0.7,
    "max_tokens": 512,
    "stream": False
}

inference_response = requests.post(
    f"{COMPUTE_ENGINE_URL}/job/{JOB_ID}/proxy/inference/chat/completions",
    headers=headers,
    json=inference_payload
)

if inference_response.status_code == 200:
    result = inference_response.json()
    generated_text = result['choices'][0]['message']['content']
    print(f"\nGenerated text:\n{generated_text}")
    
    # Print usage stats
    usage = result.get('usage', {})
    print(f"\nUsage stats:")
    print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
    print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
    print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
else:
    print(f"Inference failed: {inference_response.text}")
```

## API Reference

### Service Job Endpoints

Your service job exposes these main endpoints through the Compute Engine proxy:

#### Training Endpoints
- `POST /train/`: Start a new training job
- `GET /train/{training_job_id}`: Get training job status
- `GET /train/{training_job_id}/events`: Stream real-time training events (SSE)
- `GET /train/{training_job_id}/events/history`: Get history of training events

#### Inference Endpoints
- `GET /inference/models`: List available models
- `POST /inference/chat/completions`: Generate text using a model (OpenAI-compatible)

### Compute Engine Endpoints

The Compute Engine provides these endpoints for interacting with your service job:

- `POST /job/{job_id}`: Submit a job
- `GET /job/{job_id}`: Get job status and results
- `GET /job/{job_id}/logs`: Stream container logs
- `GET /job/{job_id}/artifacts/{artifact_id}`: Download job artifacts
- `ANY /job/{job_id}/proxy/{path}`: Proxy requests to the service job

## Deployment

To deploy your service job:

1. Build and export the Docker image:
   ```bash
   ./scripts/image-build.sh
   ./scripts/image-export.sh
   ```
   
   The export script will automatically compress the image and calculate its SHA256 hash. You'll need this hash for on-chain registration.
   > **Note**: When using the hash for on-chain registration, you must prefix it with `0x`.

2. Upload the image to a publicly accessible URL (e.g., HTTPS storage, S3, or IPFS).

3. Register the Compute Instruction on-chain by calling the `addComputeInstruction` function:
   - [Testnet addComputeInstruction](https://moksha.vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy&source_address=0x388E3b7cAD1ff4E69d70DA820b5095FdF4c98C8b#0x248e02a6)
   - [Mainnet addComputeInstruction](https://vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy&source_address=0x388E3b7cAD1ff4E69d70DA820b5095FdF4c98C8b#0x248e02a6)
   
   Parameters:
   - `url`: The publicly accessible URL where you uploaded the image
   - `hash`: The SHA256 hash from step 1 (prefixed with `0x`)
   
   After the transaction is complete, check the transaction logs to retrieve the instruction ID that was created. You'll need this ID for the next step.

4. Get approval from the DLP owner. The DLP owner must call the `updateComputeInstruction` function to approve your instruction for their DLP:
   - [Testnet updateComputeInstruction](https://moksha.vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy&source_address=0x388E3b7cAD1ff4E69d70DA820b5095FdF4c98C8b#0x79fd41f6)
   - [Mainnet updateComputeInstruction](https://vanascan.io/address/0x5786B12b4c6Ba2bFAF0e77Ed30Bf6d32805563A5?tab=read_write_proxy&source_address=0x388E3b7cAD1ff4E69d70DA820b5095FdF4c98C8b#0x79fd41f6)
   
   Parameters:
   - `instructionId`: The instruction ID from step 3
   - `dlpId`: The ID of the DLP with the data you want to access
   - `approved`: Set to `true` to approve the instruction

5. Register the job on-chain. For long-running service jobs like this, use the `submitJobWithTee` function to assign it to a dedicated instance:
   - [Testnet submitJobWithTee](https://moksha.vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy&source_address=0x1Eb8bb29B9FFAD034b7036cFFb8FA7Be2B01182a#0xabc7728d)
   - [Mainnet submitJobWithTee](https://vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy&source_address=0x1Eb8bb29B9FFAD034b7036cFFb8FA7Be2B01182a#0xabc7728d)
   
   Parameters:
   - `maxTimeout`: Timeout in seconds. For long-running services, use `1208925819614629174706175` (maximum value)
   - `gpuRequired`: True / false to specify whether the compute job needs GPU access or not.
   - `computeInstructionId`: The instruction ID from step 3
   - `teeAddress`: The address of the TEE you want to use
   
   > **Note**: Each TEE has a configured amount of GPUs that may be assignable to jobs. Once these are exhausted, new job submissions will fail.
   
   Alternatively, for shorter jobs that don't need a dedicated TEE, you can use the `submitJob` function:
   - [Testnet submitJob](https://moksha.vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy&source_address=0x1Eb8bb29B9FFAD034b7036cFFb8FA7Be2B01182a#0xe158711b)
   - [Mainnet submitJob](https://vanascan.io/address/0xb2BFe33FA420c45F1Cf1287542ad81ae935447bd?tab=read_write_proxy)
   
   After submitting the job, check the transaction logs to retrieve the job ID. You'll need this ID for the next step. You can query the job info with the `jobs` function to see which TEE pool contract and TEE address it was assigned to.

6. Submit the job through the Compute Engine API. You must submit the job to the same TEE that was assigned in step 5
   
   > **Important**: If you submit the job to a different TEE than the one it was assigned to in step 5, the submission will be rejected.
