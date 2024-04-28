import torch


def get_torch_device():
    # Check for CUDA device
    if torch.cuda.is_available():
        return torch.device('cuda')

    # Check for MPS device on macOS
    elif torch.backends.mps.is_available():
        return torch.device('mps')

    # Default to CPU if neither CUDA nor MPS is available
    else:
        return torch.device('cpu')


# Get the device and print the result
device = get_torch_device()
print(f'Using device: {device}')
