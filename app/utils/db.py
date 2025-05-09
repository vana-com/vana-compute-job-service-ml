import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import os
import logging

from config import WORKING_DIR, settings
from app.routers.training import QueryParams

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_connection(db_path: Path):
    """Get a connection to the SQLite database."""
    if not db_path.exists():
        raise FileNotFoundError(f"Database file not found at {db_path}")
    
    return sqlite3.connect(db_path)

def execute_query(query_id: str, query: str, params: List[Any] = None) -> List[Tuple]:
    """
    Execute a query against the database.
    
    Args:
        query_id: DB ID to execute query against
        query: SQL query to execute
        params: Parameters for the query
        
    Returns:
        List of tuples containing the query results
    """
    db_path = WORKING_DIR / f"{query_id}.db"
    try:
        conn = get_connection(db_path)
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

def get_training_data(query_id: str) -> List[Dict[str, Any]]:
    """
    Extract training data from the referenced input database.
    
    Args:
        query_id: DB ID to get training data for (corresponds to query_id)
    
    Returns:
        List[Dict[str, Any]]: A list of training examples
    """
    db_path = WORKING_DIR / f"{query_id}.db"
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        
        logger.info(f"Retrieving training data from DB: {query_id}")
        
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