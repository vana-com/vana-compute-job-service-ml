import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any
import torch
from datetime import datetime
import asyncio

# Import Unsloth for efficient fine-tuning
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import TrainerCallback
from trl import SFTTrainer, SFTConfig

from config import settings
from utils.db import get_training_data, format_training_examples, save_training_status
from utils.events import add_training_event
from utils.devices import supported_dtype
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingProgressCallback(TrainerCallback):
    """Custom callback to track training progress and emit events."""
    
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.start_time = time.time()
        self.step_times = []
        self.last_log_time = 0
        self.log_interval = 5  # seconds
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        # Use synchronous function or queue the event
        event_data = {
            "message": "Training started",
            "total_steps": state.max_steps,
            "timestamp": datetime.now().isoformat()
        }
        # Non-blocking call to add event
        asyncio.create_task(add_training_event(self.job_id, "start", event_data))
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        """Called at the end of each step."""
        current_time = time.time()
        
        # Calculate step time
        if len(self.step_times) > 0:
            step_time = current_time - self.step_times[-1] if self.step_times else 0
            self.step_times.append(current_time)
        else:
            step_time = 0
            self.step_times.append(current_time)
        
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
                "timestamp": datetime.now().isoformat()
            }
            asyncio.create_task(add_training_event(self.job_id, "progress", event_data))
        
        return control
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logs are available."""
        if logs:
            event_data = {
                "step": state.global_step,
                "logs": logs,
                "timestamp": datetime.now().isoformat()
            }
            asyncio.create_task(add_training_event(self.job_id, "log", event_data))
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        event_data = {
            "message": "Training completed",
            "total_steps": state.global_step,
            "total_time": round(time.time() - self.start_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        asyncio.create_task(add_training_event(self.job_id, "complete", event_data))
        return control

async def train_model(
    job_id: str,
    query_id: str,
    model_name: str,
    output_dir: Path,
    training_params: Dict[str, Any]
):
    """
    Train a model using Unsloth with data from a specific query.
    
    Args:
        job_id: Unique identifier for the training job
        query_id: ID of the query to use for training data
        model_name: Name of the base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        training_params: Parameters for training
    """
    try:
        # Update status to "preparing"
        save_training_status(job_id, {
            "job_id": job_id,
            "query_id": query_id,
            "status": "preparing",
            "message": "Preparing for training",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "output_dir": str(output_dir)
        })
        
        # Emit initial event
        await add_training_event(
            job_id, 
            "init", 
            {
                "job_id": job_id,
                "query_id": query_id,
                "model_name": model_name,
                "output_dir": str(output_dir),
                "training_params": training_params,
                "timestamp": datetime.now().isoformat()
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
        
        raw_data = get_training_data(query_id)
        
        if not raw_data or len(raw_data) == 0:
            error_msg = "No training data found in query results"
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
        training_examples = format_training_examples(raw_data)
        
        if not training_examples or len(training_examples) == 0:
            error_msg = "Failed to format training examples"
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
            "output_dir": str(output_dir),
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
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=training_params.get("max_seq_length", settings.MAX_SEQ_LENGTH),
            dtype=supported_dtype,
            load_in_4bit=True
        )
        
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
        
        # Convert training examples to dataset
        def formatting_func(examples):
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
        train_dataset = Dataset.from_list(training_examples)
        
        # Update status to "training"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "training",
            "message": "Training in progress",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "query_id": query_id,
            "num_examples": len(training_examples)
        })
        
        # Set up training arguments
        logger.info("Setting up training arguments")
        num_epochs = training_params.get("num_epochs", settings.NUM_EPOCHS)
        batch_size = training_params.get("batch_size", settings.BATCH_SIZE)
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "training_setup",
                "message": "Setting up training arguments",
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": training_params.get("learning_rate", settings.LEARNING_RATE),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        training_args = SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=training_params.get("learning_rate", settings.LEARNING_RATE),
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
        
        # Create progress callback
        progress_callback = TrainingProgressCallback(job_id)
        
        # Start training
        logger.info("Creating trainer")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            formatting_func=formatting_func,
            args=training_args,
            callbacks=[progress_callback]
        )
        
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
        
        trainer.train()
        
        # Save the model
        logger.info(f"Training complete, saving model to {output_dir}")
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "saving_model",
                "message": f"Training complete, saving model to {output_dir}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "base_model": model_name,
            "created_at": datetime.now().isoformat(),
            "training_params": training_params,
            "query_id": query_id,
            "num_examples": len(training_examples)
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model metadata saved to {output_dir}/metadata.json")
        
        # Update status to "completed"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "completed",
            "message": "Training completed successfully",
            "start_time": datetime.now().isoformat(),
            "end_time": datetime.now().isoformat(),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "query_id": query_id,
            "num_examples": len(training_examples)
        })
        
        await add_training_event(
            job_id, 
            "status", 
            {
                "status": "completed",
                "message": "Training completed successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"Training job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Training job {job_id} failed: {str(e)}")
        # Update status to "failed"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "failed",
            "message": f"Training failed: {str(e)}",
            "error": str(e),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "query_id": query_id
        })
        
        # Emit error event
        await add_training_event(
            job_id, 
            "error", 
            {
                "message": f"Training failed: {str(e)}",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        raise e