from typing import Tuple
import torch
from .operator import square_distance


def k_nearest_neighbors(
    coords1: torch.Tensor, coords2: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute k nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (B, C, N)
        coords2: coordinates of all points (B, C, N)
        k: number of nearest neighbors

    Returns:
        idxs: indices of k nearest neighbors (B, N, k)
        square distances: square distance for kNN (B, N, k)
    """

    # compute distances between coords1 and coords2.
    point_pairwise_distances = square_distance(
        coords1, coords2
    )  # ((B, C, N), (B, C, M)) -> (B, N, M)

    # get top-k indices and point_pairwise_distances.
    square_dists, idxs = torch.topk(
        point_pairwise_distances, k, dim=2, largest=False, sorted=True
    )  # (B, N, M) -> ((B, N, k), (B, N, k))

    return idxs, square_dists


def radius_and_k_nearest_neighbors(
    coords1: torch.Tensor, coords2: torch.Tensor, k: int, r: float
):
    """Compute radius and k nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (B, C, N)
        coords2: coordinates of all points (B, C, M)
        k: number of nearest neighbors
        r: radius

    Returns:
        idxs: indices of neighbors (B, N, k)
        square distances: square distance between pairwise points (B, N, k)
    """

    # compute kNN.
    idxs, square_dists = k_nearest_neighbors(coords1, coords2, k)

    # get radius nearest neighbors mask.
    radius_masks = square_dists < r**2

    # radius mask
    idxs[radius_masks == False] = len(coords2)
    square_dists[radius_masks == False] = -1

    return idxs, square_dists
