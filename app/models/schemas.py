from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union, Literal

class Message(BaseModel):
    """OpenAI-style chat message."""
    role: Literal["system", "user", "assistant", "function", "tool"] = "user"
    content: str
    name: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "What is machine learning?"
            }
        }

class ChatCompletionRequest(BaseModel):
    """OpenAI-style chat completion request."""
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    n: Optional[int] = 1
    max_tokens: Optional[int] = 512
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "model": "my_finetuned_model",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is machine learning?"}
                ],
                "temperature": 0.7,
                "max_tokens": 512,
                "stream": False
            }
        }

class ChatCompletionResponseChoice(BaseModel):
    """OpenAI-style chat completion response choice."""
    index: int
    message: Message
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponseUsage(BaseModel):
    """OpenAI-style token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionResponse(BaseModel):
    """OpenAI-style chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: ChatCompletionResponseUsage

class ChatCompletionChunkChoice(BaseModel):
    """OpenAI-style streaming chat completion chunk choice."""
    index: int
    delta: Dict[str, Any]
    finish_reason: Optional[str] = None

class ChatCompletionChunk(BaseModel):
    """OpenAI-style streaming chat completion chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]