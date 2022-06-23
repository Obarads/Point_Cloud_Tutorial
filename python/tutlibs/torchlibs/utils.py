import numpy as np
import torch


def t2n(torch_tensor: torch.Tensor) -> np.ndarray:
    """torch.Tensor to numpy.ndarray"""
    return torch_tensor.detach().cpu().numpy()
