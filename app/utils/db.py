import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Union
import json
import os
import logging

from app.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection():
    """Get a connection to the SQLite database."""
    if not settings.DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found at {settings.DB_PATH}")
    
    return sqlite3.connect(settings.DB_PATH)

def execute_query(query: str, params: List[Any] = None) -> List[Tuple]:
    """
    Execute a query against the database.
    
    Args:
        query: SQL query to execute
        params: Parameters for the query
        
    Returns:
        List of tuples containing the query results
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        results = cursor.fetchall()
        conn.close()
        return results
    
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise Exception(f"Failed to execute query: {str(e)}")

def get_training_data(query_info: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Extract training data from the input database based on query information.
    
    Args:
        query_info: Dictionary containing query information. Can be:
            - {"query_id": str} - ID of an existing query
            - {"query": str, "params": List, "refiner_id": int, "query_signature": str} - Parameters for a new query
            - None - Use default query to get all data from results table
    
    Returns:
        List[Dict[str, Any]]: A list of training examples
    """
    try:
        conn = get_connection()
        cursor = conn.cursor()
        
        # Log query information
        if query_info:
            logger.info(f"Using query info: {query_info}")
        else:
            logger.info("No query info provided, using default query")
        
        # If query_info is provided and contains a query, execute it
        # In a real implementation, this would interact with the Query Engine
        # For now, we'll just log it and proceed with the default query
        if query_info and "query" in query_info:
            logger.info(f"Would execute query: {query_info['query']} with params: {query_info.get('params', [])}")
            logger.info(f"Refiner ID: {query_info.get('refiner_id')}, Query Signature: {query_info.get('query_signature')}")
            
            # In a real implementation, this would be:
            # 1. Submit the query to the Query Engine
            # 2. Wait for the Query Engine to process it
            # 3. Download the results to the input directory
            # 4. Use the results for training
            
            # For now, we'll just log that we would do this
            logger.info("In production, this would submit the query to the Query Engine")
            logger.info("For now, using existing query_results.db")
        
        # Get the schema of the results table
        cursor.execute("PRAGMA table_info(results)")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Query all data from the results table
        cursor.execute("SELECT * FROM results")
        rows = cursor.fetchall()
        
        # Convert rows to list of dictionaries
        data = []
        for row in rows:
            item = {}
            for i, col in enumerate(columns):
                item[col] = row[i]
            data.append(item)
        
        conn.close()
        
        # Log the number of examples found
        logger.info(f"Found {len(data)} examples in the database")
        
        return data
    
    except Exception as e:
        logger.error(f"Error getting training data: {e}")
        raise Exception(f"Failed to extract training data: {str(e)}")

def format_training_examples(data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Format raw data into training examples for the model.
    
    This function converts the raw data from the database into the format
    expected by the model for training.
    
    Args:
        data: Raw data from the database
        
    Returns:
        List of formatted training examples
    """
    examples = []
    
    # This implementation will adapt based on the structure of the data
    # It tries to intelligently format the data for training
    for item in data:
        # Check for common field patterns
        
        # Case 1: Text and response fields (ideal for chat models)
        if "text" in item and "response" in item:
            examples.append({
                "prompt": f"<s>[INST] {item['text']} [/INST]",
                "completion": item['response']
            })
        
        # Case 2: Question and answer fields
        elif "question" in item and "answer" in item:
            examples.append({
                "prompt": f"<s>[INST] {item['question']} [/INST]",
                "completion": item['answer']
            })
        
        # Case 3: Content and summary fields (for summarization)
        elif "content" in item and "summary" in item:
            examples.append({
                "prompt": f"<s>[INST] Summarize the following text: {item['content']} [/INST]",
                "completion": item['summary']
            })
        
        # Case 4: User and locale fields (from the example data)
        elif "user_id" in item and "locale" in item:
            examples.append({
                "prompt": f"<s>[INST] What is the locale for user {item['user_id']}? [/INST]",
                "completion": f"The locale for user {item['user_id']} is {item['locale']}."
            })
        
        # Case 5: Single text field (for text completion)
        elif "text" in item:
            examples.append({
                "prompt": f"<s>[INST] Generate a response for: {item['text']} [/INST]",
                "completion": "This is a placeholder response."
            })
        
        # Case 6: Fallback for other field combinations
        else:
            # Create a prompt from all fields
            prompt_text = "Information: " + ", ".join([f"{k}: {v}" for k, v in item.items()])
            examples.append({
                "prompt": f"<s>[INST] {prompt_text} [/INST]",
                "completion": "This is a placeholder response based on the provided information."
            })
    
    logger.info(f"Formatted {len(examples)} training examples")
    return examples

def save_training_status(job_id: str, status: Dict[str, Any]):
    """
    Save the training status to a JSON file.
    
    Args:
        job_id: The ID of the training job
        status: The status information to save
    """
    status_file = settings.WORKING_DIR / f"{job_id}_status.json"
    
    with open(status_file, "w") as f:
        json.dump(status, f, indent=4)
    
    logger.info(f"Saved training status for job {job_id}: {status['status']}")