

def get_model_path(model_name: str) -> Path:
    """
    Get the path to a model directory.
    
    Args:
        model_name: Name of the model to find
        
    Returns:
        Path to the model directory
        
    Raises:
        HTTPException: If the model is not found
    """
    model_path = settings.OUTPUT_DIR / model_name
    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model not found: {model_name}"
        )
    return model_path

def generate_completion_id() -> str:
    """
    Generate a unique ID for a completion.
    
    Returns:
        Unique completion ID string
    """
    return f"chatcmpl-{uuid.uuid4().hex}"

def get_current_timestamp() -> int:
    """
    Get the current Unix timestamp.
    
    Returns:
        Current time as Unix timestamp (integer)
    """
    return int(time.time())


def load_model_metadata(model_dir: Path) -> Dict[str, Any]:
    """
    Load metadata for a model from its metadata.json file.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Dictionary containing model metadata, or empty dict if not found
    """
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        return {}
        
    try:
        with open(metadata_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def parse_model_timestamp(timestamp_str: str) -> int:
    """
    Parse a timestamp string into a Unix timestamp.
    
    Args:
        timestamp_str: Timestamp string in ISO format
        
    Returns:
        Unix timestamp as integer
    """
    try:
        return int(time.mktime(time.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")))
    except Exception:
        return get_current_timestamp()

def create_model_data(model_dir: Path) -> ModelData:
    """
    Create a ModelData object for a model directory.
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        ModelData object with model information
    """
    metadata = load_model_metadata(model_dir)
    
    # Get created timestamp
    created = get_current_timestamp()
    if "created_at" in metadata:
        created = parse_model_timestamp(metadata["created_at"])
    
    # Get root model name
    root = metadata.get("base_model", "unknown")
    
    return ModelData(
        id=model_dir.name,
        created=created,
        root=root
    )
