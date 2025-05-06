import json
import time
import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
from collections import defaultdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory storage for training events
# This could be replaced with Redis or another distributed store in production
training_events = defaultdict(list)
training_event_subscribers = defaultdict(set)

async def add_training_event(job_id: str, event_type: str, data: Dict[str, Any]):
    """
    Add a training event to the event store.
    
    Args:
        job_id: The ID of the training job
        event_type: The type of event (e.g., "progress", "error", "complete")
        data: The event data
    """
    event = {
        "type": event_type,
        "timestamp": time.time(),
        "data": data
    }
    
    # Add event to storage
    training_events[job_id].append(event)
    
    # Log the event
    logger.info(f"Training event for job {job_id}: {event_type}")
    
    # Notify subscribers
    for queue in training_event_subscribers[job_id]:
        await queue.put(event)

async def get_training_events(job_id: str) -> List[Dict[str, Any]]:
    """
    Get all training events for a job.
    
    Args:
        job_id: The ID of the training job
        
    Returns:
        List of training events
    """
    return training_events.get(job_id, [])

async def subscribe_to_training_events(job_id: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Subscribe to training events for a job.
    
    Args:
        job_id: The ID of the training job
        
    Yields:
        Training events as they occur
    """
    # Create a queue for this subscriber
    queue = asyncio.Queue()
    
    # Add to subscribers
    training_event_subscribers[job_id].add(queue)
    
    try:
        # Send all existing events first
        for event in training_events.get(job_id, []):
            yield event
        
        # Then wait for new events
        while True:
            event = await queue.get()
            yield event
            
            # If this is a completion or error event, stop streaming
            if event["type"] in ["complete", "error"]:
                break
    finally:
        # Remove subscriber when done
        training_event_subscribers[job_id].discard(queue)
        
        # If no more subscribers, clean up
        if not training_event_subscribers[job_id]:
            del training_event_subscribers[job_id]

def format_sse_event(event: Dict[str, Any]) -> str:
    """
    Format an event as an SSE message.
    
    Args:
        event: The event to format
        
    Returns:
        SSE-formatted string
    """
    event_str = f"event: {event['type']}\n"
    event_str += f"data: {json.dumps(event['data'])}\n\n"
    return event_str