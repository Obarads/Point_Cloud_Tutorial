import torch
from .backend import _backend
from typing import Tuple


def k_nearest_neighbor(
    center_coords: torch.Tensor, coords: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute k nearest neighbors between coords and center_coords.

    Args:
        center_coords: xyz of center points [B, C, M]
        coords: xyz of all points [B, C, N]
        k: number of nearest neighbors

    Returns:
        idxs: top k idx between coords and center_coords [B, M, k]
    """
    coords = coords.contiguous()
    center_coords = center_coords.contiguous()
    idxs, _ = _backend.k_nearest_neighbors(coords, center_coords, k)

    return idxs
