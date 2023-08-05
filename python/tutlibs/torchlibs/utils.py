import numpy as np
import torch


def t2n(torch_tensor: torch.Tensor) -> np.ndarray:
    """torch.Tensor to numpy.ndarray"""
    return torch_tensor.detach().cpu().numpy()

def torch_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
