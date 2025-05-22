import time
import json
import logging
import traceback
import os
import sys
import psutil
import torch
from pathlib import Path
from typing import Dict, Any
from datetime import datetime
import asyncio
import zipfile

# Import Unsloth for efficient fine-tuning
from unsloth import FastLanguageModel, __version__ as unsloth_version
from datasets import Dataset, __version__ as datasets_version
from transformers import TrainerCallback, __version__ as transformers_version
from trl import SFTTrainer, SFTConfig, __version__ as trl_version

from app.config import settings
from app.utils.db import get_training_data, format_training_examples, save_training_status
from app.utils.events import add_training_event
from app.utils.devices import supported_dtype, get_device

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to track training progress and emit events."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        self.step_times = []
        self.last_log_time = 0
        self.log_interval = 5  # seconds
        self.memory_usage = []
        self.gpu_usage = []
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        try:
            # Log system info
            logger.info(f"Training job {self.job_id} started")
            logger.info(f"Training parameters: {args.to_dict()}")
            logger.info(f"Max steps: {state.max_steps}")
            
            # Record initial resource usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.memory_usage.append(memory_info.rss / 1024 / 1024)  # MB
            
            # GPU info if available
            gpu_info = get_device() if torch.cuda.is_available() else "No GPU available"
            logger.info(f"GPU info: {gpu_info}")
            
            # Use synchronous function or queue the event
            event_data = {
                "message": "Training started",
                "total_steps": state.max_steps,
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "memory_mb": self.memory_usage[-1],
                    "gpu_info": gpu_info
                }
            }
            # Non-blocking call to add event
            asyncio.create_task(add_training_event(self.job_id, "start", event_data))
        except Exception as e:
            logger.error(f"Error in on_train_begin: {str(e)}")
            logger.error(traceback.format_exc())
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        try:
            current_time = time.time()
            
            # Calculate step time
            if len(self.step_times) > 0:
                step_time = current_time - self.step_times[-1] if self.step_times else 0
                self.step_times.append(current_time)
            else:
                step_time = 0
                self.step_times.append(current_time)
            
            # Record resource usage periodically
            if current_time - self.last_log_time >= self.log_interval or state.global_step == state.max_steps:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024  # MB
                self.memory_usage.append(memory_mb)
                
                # GPU memory if available
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                    self.gpu_usage.append({
                        "allocated_mb": gpu_memory_allocated,
                        "reserved_mb": gpu_memory_reserved
                    })
                    logger.debug(f"Step {state.global_step}: Memory usage: {memory_mb:.2f} MB, GPU allocated: {gpu_memory_allocated:.2f} MB")
                else:
                    logger.debug(f"Step {state.global_step}: Memory usage: {memory_mb:.2f} MB")
            
            # Only emit events periodically to avoid overwhelming the client
            if current_time - self.last_log_time >= self.log_interval or state.global_step == state.max_steps:
                self.last_log_time = current_time
                
                # Calculate progress percentage
                progress = (state.global_step / state.max_steps) * 100 if state.max_steps > 0 else 0
                
                # Calculate ETA
                if len(self.step_times) > 1:
                    avg_step_time = (self.step_times[-1] - self.step_times[0]) / (len(self.step_times) - 1)
                    steps_remaining = state.max_steps - state.global_step
                    eta_seconds = avg_step_time * steps_remaining
                    eta = str(datetime.fromtimestamp(current_time + eta_seconds).strftime('%Y-%m-%d %H:%M:%S'))
                else:
                    eta = "calculating..."
                
                # Resource info
                resource_info = {
                    "memory_mb": self.memory_usage[-1] if self.memory_usage else None,
                }
                if torch.cuda.is_available() and self.gpu_usage:
                    resource_info["gpu"] = self.gpu_usage[-1]
                
                # Queue event in non-blocking way
                event_data = {
                    "step": state.global_step,
                    "total_steps": state.max_steps,
                    "progress": round(progress, 2),
                    "loss": state.log_history[-1]["loss"] if state.log_history else None,
                    "learning_rate": state.log_history[-1]["learning_rate"] if state.log_history else None,
                    "step_time": round(step_time, 2),
                    "elapsed": round(current_time - self.start_time, 2),
                    "eta": eta,
                    "timestamp": datetime.now().isoformat(),
                    "resources": resource_info
                }
                
                # Log progress at INFO level
                if state.global_step % 10 == 0 or state.global_step == state.max_steps:
                    logger.info(f"Step {state.global_step}/{state.max_steps} ({progress:.2f}%): Loss: {event_data['loss']}, ETA: {eta}")
                
                asyncio.create_task(add_training_event(self.job_id, "progress", event_data))
        except Exception as e:
            logger.error(f"Error in on_step_end at step {state.global_step}: {str(e)}")
            logger.error(traceback.format_exc())
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are available."""
        try:
            if logs:
                event_data = {
                    "step": state.global_step,
                    "logs": logs,
                    "timestamp": datetime.now().isoformat()
                }
                asyncio.create_task(add_training_event(self.job_id, "log", event_data))
        except Exception as e:
            logger.error(f"Error in on_log at step {state.global_step}: {str(e)}")
            logger.error(traceback.format_exc())
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        try:
            # Final resource usage
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # MB
            
            # GPU memory if available
            gpu_info = None
            if torch.cuda.is_available():
                gpu_memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                gpu_info = {
                    "allocated_mb": gpu_memory_allocated,
                    "reserved_mb": gpu_memory_reserved
                }
                
            total_time = round(time.time() - self.start_time, 2)
            logger.info(f"Training completed in {total_time} seconds")
            logger.info(f"Final memory usage: {memory_mb:.2f} MB")
            if gpu_info:
                logger.info(f"Final GPU memory: allocated={gpu_info['allocated_mb']:.2f}MB, reserved={gpu_info['reserved_mb']:.2f}MB")
            
            event_data = {
                "message": "Training completed",
                "total_steps": state.global_step,
                "total_time": total_time,
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "memory_mb": memory_mb,
                    "gpu_info": gpu_info
                }
            }
            asyncio.create_task(add_training_event(self.job_id, "complete", event_data))
        except Exception as e:
            logger.error(f"Error in on_train_end: {str(e)}")
            logger.error(traceback.format_exc())
        return control

async def train_model(
    job_id: str,
    query_id: str,
    model_name: str,
    working_path: Path,
    output_path: Path,
    training_params: Dict[str, Any]
):
    """
    Train a model using Unsloth with data from a specific query.
    
    Args:
        job_id: Unique identifier for the training job
        query_id: ID of the query to use for training data
        model_name: Name of the base model to fine-tune
        working_path: Directory path to save the fine-tuned model to
        output_path: Zip file path to save the fine-tuned model to for artifact scanning + download
        training_params: Parameters for training
    """
    start_time = time.time()
    
    # Log library versions for debugging compatibility issues
    logger.info(f"Starting training with library versions:")
    logger.info(f"- Python: {sys.version}")
    logger.info(f"- PyTorch: {torch.__version__}")
    logger.info(f"- Unsloth: {unsloth_version}")
    logger.info(f"- Transformers: {transformers_version}")
    logger.info(f"- Datasets: {datasets_version}")
    logger.info(f"- TRL: {trl_version}")
    
    # Log GPU information
    if torch.cuda.is_available():
        logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"Device capabilities: {torch.cuda.get_device_capability(0)}")
        logger.info(f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024:.2f} GB")
    else:
        logger.warning("No GPU available! Training will be very slow. CPU-only training is not recommended.")
    
    try:
        # Log training parameters
        logger.info(f"Training job {job_id} configuration:")
        logger.info(f"- Model name: {model_name}")
        logger.info(f"- Working directory: {working_path}")
        logger.info(f"- Artifact path: {output_path}")
        logger.info(f"- Query ID: {query_id}")
        logger.info(f"- Training parameters: {training_params}")
        
        # Update status to "preparing"
        save_training_status(job_id, {
            "job_id": job_id,
            "query_id": query_id,
            "status": "preparing",
            "message": "Preparing for training",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "working_path": str(working_path),
            "output_path": str(output_path)
        })
        
        # Emit initial event
        await add_training_event(
            job_id, 
            "init", 
            {
                "job_id": job_id,
                "query_id": query_id,
                "model_name": model_name,
                "working_path": str(working_path),
                "output_path": str(output_path),
                "training_params": training_params,
                "timestamp": datetime.now().isoformat(),
                "system_info": {
                    "python_version": sys.version,
                    "pytorch_version": torch.__version__,
                    "unsloth_version": unsloth_version,
                    "transformers_version": transformers_version,
                    "datasets_version": datasets_version,
                    "trl_version": trl_version,
                    "has_gpu": torch.cuda.is_available()
                }
            }
        )
        
        logger.info(f"Starting training job {job_id} with model {model_name}")
        logger.info(f"Query id: {query_id}")
        
        # Get training data using the query information
        logger.info("Fetching training data from query results")
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "fetching_data",
                "message": "Fetching training data from query results",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            raw_data = get_training_data(query_id)
            logger.info(f"Successfully fetched {len(raw_data) if raw_data else 0} training examples")
        except Exception as e:
            error_msg = f"Failed to fetch training data: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise ValueError(error_msg) from e
        
        if not raw_data or len(raw_data) == 0:
            error_msg = "No training data found in query results"
            logger.error(error_msg)
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise ValueError(error_msg)
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "formatting_data",
                "message": f"Formatting {len(raw_data)} examples for training",
                "num_examples": len(raw_data),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Formatting {len(raw_data)} examples for training")
        try:
            training_examples = format_training_examples(raw_data)
            logger.info(f"Successfully formatted {len(training_examples)} training examples")
        except Exception as e:
            error_msg = f"Failed to format training examples: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise ValueError(error_msg) from e
        
        if not training_examples or len(training_examples) == 0:
            error_msg = "Failed to format training examples - no valid examples produced"
            logger.error(error_msg)
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise ValueError(error_msg)
        
        # Update status to "loading_model"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "loading_model",
            "message": f"Loading model {model_name}",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "working_path": str(working_path),
            "output_path": str(output_path),
            "query_id": query_id,
            "num_examples": len(training_examples)
        })
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "loading_model",
                "message": f"Loading model {model_name}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Loading model {model_name}")
        # Load model with Unsloth
        try:
            max_seq_length = training_params.get("max_seq_length", settings.MAX_SEQ_LENGTH)
            logger.info(f"Using max_seq_length: {max_seq_length}, dtype: {supported_dtype}")
            
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_name,
                max_seq_length=max_seq_length,
                dtype=supported_dtype,
                load_in_4bit=True
            )
            logger.info(f"Successfully loaded model {model_name}")
            logger.info(f"Model size: {model.get_memory_footprint() / 1024 / 1024 / 1024:.2f} GB")
        except Exception as e:
            error_msg = f"Failed to load model {model_name}: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "model": model_name,
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "preparing_model",
                "message": "Preparing model for training with LoRA",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info("Preparing model for training with LoRA")
        # Prepare model for training
        try:
            model = FastLanguageModel.get_peft_model(
                model,
                r=16,  # LoRA rank
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
            )
            logger.info("Successfully applied LoRA adapters to model")
        except Exception as e:
            error_msg = f"Failed to prepare model with LoRA: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        # Convert training examples to dataset
        def formatting_func(examples):
            try:
                # Handle both single examples and batches
                if isinstance(examples["prompt"], str):
                    # Single example case - must return a list with one string
                    return [examples["prompt"] + examples["completion"] + tokenizer.eos_token]
                else:
                    # Batch case
                    output_texts = []
                    for i in range(len(examples["prompt"])):
                        text = examples["prompt"][i] + examples["completion"][i] + tokenizer.eos_token
                        output_texts.append(text)
                    return output_texts
            except Exception as e:
                logger.error(f"Error in formatting_func: {str(e)}")
                logger.error(f"Input examples: {examples}")
                logger.error(traceback.format_exc())
                raise e
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "creating_dataset",
                "message": "Creating training dataset",
                "num_examples": len(training_examples),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info("Creating training dataset")
        try:
            train_dataset = Dataset.from_list(training_examples)
            logger.info(f"Created dataset with {len(train_dataset)} examples")
            
            # Sample and log a few examples for debugging
            sample_size = min(3, len(train_dataset))
            sample_indices = list(range(sample_size))
            samples = train_dataset.select(sample_indices)
            
            logger.info(f"Sample training examples (first {sample_size}):")
            for i, example in enumerate(samples):
                prompt_preview = example["prompt"][:100] + "..." if len(example["prompt"]) > 100 else example["prompt"]
                completion_preview = example["completion"][:100] + "..." if len(example["completion"]) > 100 else example["completion"]
                logger.info(f"Example {i+1}:")
                logger.info(f"  Prompt: {prompt_preview}")
                logger.info(f"  Completion: {completion_preview}")
        except Exception as e:
            error_msg = f"Failed to create dataset: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        # Update status to "training"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "training",
            "message": "Training in progress",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "working_path": str(working_path),
            "output_path": str(output_path),
            "query_id": query_id,
            "num_examples": len(training_examples)
        })
        
        # Set up training arguments
        logger.info("Setting up training arguments")
        num_epochs = training_params.get("num_epochs", settings.NUM_EPOCHS)
        batch_size = training_params.get("batch_size", settings.BATCH_SIZE)
        learning_rate = training_params.get("learning_rate", settings.LEARNING_RATE)
        
        logger.info(f"Training parameters: epochs={num_epochs}, batch_size={batch_size}, learning_rate={learning_rate}")
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "training_setup",
                "message": "Setting up training arguments",
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            training_args = SFTConfig(
                output_dir=str(working_path),
                num_train_epochs=num_epochs,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="paged_adamw_32bit",
                weight_decay=0.01,
                lr_scheduler_type="cosine",
                warmup_ratio=0.03,
                save_strategy="epoch",
                report_to="none"
            )
            logger.info(f"Training config created: {training_args}")
        except Exception as e:
            error_msg = f"Failed to create training configuration: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        # Create progress callback
        progress_callback = TrainingProgressCallback(job_id)
        
        # Start training
        logger.info("Creating trainer")
        try:
            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                formatting_func=formatting_func,
                args=training_args,
                callbacks=[progress_callback]
            )
            logger.info("Successfully created SFTTrainer")
        except Exception as e:
            error_msg = f"Failed to create trainer: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        # Train the model
        logger.info("Starting training")
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "training_started",
                "message": "Starting training process",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            trainer.train()
            logger.info("Training completed successfully")
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            # Get detailed information about the error
            gpu_info = get_device() if torch.cuda.is_available() else "No GPU"
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024  # MB
            
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "system_info": {
                        "memory_mb": memory_mb,
                        "gpu_info": gpu_info
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
            
            # Update status to "failed"
            save_training_status(job_id, {
                "job_id": job_id,
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__,
                "model_name": model_name,
                "working_path": str(working_path),
                "output_path": str(output_path),
                "query_id": query_id
            })
            
            raise RuntimeError(error_msg) from e
        
        # Save the model
        logger.info(f"Training complete, saving model to {working_path}")
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "saving_model",
                "message": f"Training complete, saving model to {working_path}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        try:
            # Ensure the output directory exists
            os.makedirs(working_path, exist_ok=True)
            logger.info(f"Saving model to {working_path}")
            
            model.save_pretrained(str(working_path))
            logger.info(f"Model saved to {working_path}")
            
            tokenizer.save_pretrained(str(working_path))
            logger.info(f"Tokenizer saved to {working_path}")
        except Exception as e:
            error_msg = f"Failed to save model: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": error_msg,
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat()
                }
            )
            raise RuntimeError(error_msg) from e
        
        # Save metadata
        try:
            metadata = {
                "job_id": job_id,
                "base_model": model_name,
                "created_at": datetime.now().isoformat(),
                "training_params": training_params,
                "query_id": query_id,
                "num_examples": len(training_examples),
                "training_duration_seconds": round(time.time() - start_time, 2),
                "library_versions": {
                    "python": sys.version,
                    "pytorch": torch.__version__,
                    "unsloth": unsloth_version,
                    "transformers": transformers_version,
                    "datasets": datasets_version,
                    "trl": trl_version
                }
            }
            
            metadata_path = working_path / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=4)
            
            logger.info(f"Model metadata saved to {metadata_path}")
        except Exception as e:
            error_msg = f"Failed to save metadata: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            # This is non-fatal, so we'll continue
        
        # Once training is complete, create the zip file for artifact scanning
        if not zip_model_directory(working_path, output_path):
            logger.error(f"Failed to create zip file artifact for job {job_id}")

        # Update status to "completed"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "completed",
            "message": "Training completed successfully",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "model_name": model_name,
            "working_path": str(working_path),
            "output_path": str(output_path),
            "query_id": query_id,
            "num_examples": len(training_examples),
            "duration_seconds": round(time.time() - start_time, 2)
        })
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "completed",
                "message": "Training completed successfully",
                "duration_seconds": round(time.time() - start_time, 2),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Training job {job_id} completed successfully in {round(time.time() - start_time, 2)} seconds")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Update status to "failed"
        try:
            save_training_status(job_id, {
                "job_id": job_id,
                "status": "failed",
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "error_type": type(e).__name__,
                "model_name": model_name,
                "working_path": str(working_path),
                "output_path": str(output_path),
                "query_id": query_id,
                "duration_seconds": round(time.time() - start_time, 2)
            })
        except Exception as inner_e:
            # If we can't even save the status, just log it
            logger.error(f"Failed to save failed status: {str(inner_e)}")
        
        # Emit error event
        try:
            await add_training_event(
                job_id, 
                "error", 
                {
                    "message": f"Training failed: {str(e)}",
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "error_trace": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                    "duration_seconds": round(time.time() - start_time, 2)
                }
            )
        except Exception as event_e:
            # If we can't emit the event, just log it
            logger.error(f"Failed to emit error event: {str(event_e)}")
        
        raise e

def zip_model_directory(source_dir: Path, output_zip: Path) -> bool:
        """
        Create a zip file from the model directory.
        
        Args:
            source_dir: Path to the model directory
            output_zip: Path to the output zip file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info(f"Creating zip file of model at {source_dir} to {output_zip}")
            
            # Create parent directory if it doesn't exist
            if not output_zip.parent.exists():
                output_zip.parent.mkdir(parents=True, exist_ok=True)
            
            # If the zip file already exists, remove it
            if output_zip.exists():
                logger.info(f"Removing existing zip file: {output_zip}")
                output_zip.unlink()
            
            # Create the zip file
            with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Walk through the directory
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = Path(root) / file
                        # Calculate the relative path to maintain directory structure
                        rel_path = file_path.relative_to(source_dir.parent)
                        zipf.write(file_path, rel_path)
            
            logger.info(f"Successfully created zip file: {output_zip}")
            return True
        except Exception as e:
            logger.error(f"Failed to create zip file: {str(e)}")
            return False
