import typing as t

import torch


def get_gpu_device_safe(gpu_device: t.Union[torch.device, int]) -> torch.device:
    """
    Given a torch gpu or integer index of device, retrieve the device object if it is available.
    Otherwise, return error.
    Args:
        gpu_device: The torch gpu device object or index of gpu

    Returns:
        The torch.device object if available. Otherwise, return default cpu device.
    """
    if torch.cuda.is_available():
        device_index = gpu_device.index if isinstance(gpu_device, torch.device) else gpu_device
        if device_index < torch.cuda.device_count():
            return torch.device(f'cuda:{device_index}')
    return torch.device('cpu')