import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def is_bfloat16_device(device):
    if device.type == "cuda":
        return torch.cuda.is_bf16_supported()
    elif device.type == "mps":
        return torch.backends.mps.is_bf16_supported()
    return False

def get_supported_dtype():
    device = get_device()

    if device.type == "cpu":
        return torch.float32

    if is_bfloat16_device(device):
        return torch.bfloat16
    else:
        return torch.float16


supported_dtype = get_supported_dtype()