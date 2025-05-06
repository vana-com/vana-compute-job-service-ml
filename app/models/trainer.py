import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from datetime import datetime

# Import Unsloth for efficient fine-tuning
from unsloth import FastLanguageModel
from datasets import Dataset
import bitsandbytes as bnb
from transformers import TrainingArguments

from app.config import settings
from app.utils.db import get_training_data, format_training_examples, save_training_status

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def train_model(
    job_id: str,
    model_name: str,
    output_dir: Path,
    training_params: Dict[str, Any],
    query_info: Optional[Dict[str, Any]] = None
):
    """
    Train a model using Unsloth with data from a specific query.
    
    Args:
        job_id: Unique identifier for the training job
        model_name: Name of the base model to fine-tune
        output_dir: Directory to save the fine-tuned model
        training_params: Parameters for training
        query_info: Information about the query to use for training data
    """
    try:
        # Update status to "preparing"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "preparing",
            "message": "Preparing for training",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "query_info": query_info
        })
        
        logger.info(f"Starting training job {job_id} with model {model_name}")
        logger.info(f"Query info: {query_info}")
        
        # Get training data using the query information
        logger.info("Fetching training data from query results")
        raw_data = get_training_data(query_info)
        
        if not raw_data or len(raw_data) == 0:
            raise ValueError("No training data found in query results")
        
        logger.info(f"Formatting {len(raw_data)} examples for training")
        training_examples = format_training_examples(raw_data)
        
        if not training_examples or len(training_examples) == 0:
            raise ValueError("Failed to format training examples")
        
        # Update status to "loading_model"
        save_training_status(job_id, {
            "job_id": job_id,
            "status": "loading_model",
            "message": f"Loading model {model_name}",
            "start_time": datetime.now().isoformat(),
            "model_name": model_name,
            "output_dir": str(output_dir),
            "query_info": query_info,
            "num_examples": len(training_examples)
        })
        
        logger.info(f"Loading model {model_name}")
        # Load model with Unsloth
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=training_params.get("max_seq_length", settings.MAX_SEQ_LENGTH),
            dtype=torch.bfloat16,
            load_in_4bit=True
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
            output_texts = []
            for i in range(len(examples["prompt"])):
                text = examples["prompt"][i] + examples["completion"][i] + tokenizer.eos_token
                output_texts.append(text)
            return output_texts
        
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
            "query_info": query_info,
            "num_examples": len(training_examples)
        })
        
        # Set up training arguments
        logger.info("Setting up training arguments")
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=training_params.get("num_epochs", settings.NUM_EPOCHS),
            per_device_train_batch_size=training_params.get("batch_size", settings.BATCH_SIZE),
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
        
        # Start training
        logger.info("Creating trainer")
        trainer = FastLanguageModel.get_trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            formatting_func=formatting_func,
            args=training_args,
        )
        
        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Save the model
        logger.info(f"Training complete, saving model to {output_dir}")
        model.save_pretrained(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        
        # Save metadata
        metadata = {
            "job_id": job_id,
            "base_model": model_name,
            "created_at": datetime.now().isoformat(),
            "training_params": training_params,
            "query_info": query_info,
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
            "query_info": query_info,
            "num_examples": len(training_examples)
        })
        
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
            "query_info": query_info
        })
        raise e