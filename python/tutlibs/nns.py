from typing import Tuple
import numpy as np

from .operator import square_distance


def k_nearest_neighbors(coords1: np.ndarray, coords2: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute k nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (N, C)
        coords2: coordinates of all points (M, C)
        k: number of nearest neighbors

    Returns:
        idxs: indices of k nearest neighbors (N, k)
        square distances: square distance for kNN (N, k)
    """

    # compute distances between coords1 and coords2.
    point_pairwise_distances = square_distance(coords1, coords2) # shape: (N, M)

    # sort the distances between two points in order of closeness and get top-k indices.
    idxs = np.argsort(point_pairwise_distances, axis=-1)[:, :k] # shape: (N, k)

    # get the distance between two points according to the top-k indices.
    square_dists = np.take_along_axis(point_pairwise_distances, idxs, axis=-1) # shape: (N, k)

    return idxs, square_dists


def radius_nearest_neighbors(coords1: np.ndarray, coords2: np.ndarray, r: float, k: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute radius nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (N, C)
        coords2: coordinates of all points (M, C)
        r: radius
        k: number of nearest neighbors

    Returns:
        idxs: indices of k nearest neighbors (N, k)
        square distances: square distance for kNN (N, k)
        mask : radius mask (bool) for idxs an distance (N, k)
    """

    # compute kNN.
    idxs, square_dists = k_nearest_neighbors(coords1, coords2, k)

    # get radius nearest neighbors mask.
    radius_mask = square_dists < r**2

    return idxs, square_dists, radius_mask
