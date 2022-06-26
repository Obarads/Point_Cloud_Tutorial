from typing import List, Tuple
import numpy as np

from .operator import square_distance


def k_nearest_neighbors(
    coords1: np.ndarray, coords2: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    point_pairwise_distances = square_distance(
        coords1, coords2
    )  # ((N, 3), (M, 3)) -> (N, M)

    # sort the distances between two points in order of closeness and get top-k indices.
    idxs = np.argsort(point_pairwise_distances, axis=-1)[:, :k]  # (N, M) -> (N, k)

    # get the distance between two points according to the top-k indices.
    square_dists = np.take_along_axis(
        point_pairwise_distances, idxs, axis=-1
    )  # ((N, M), (N, k)) -> (N, k)

    return idxs, square_dists


def radius_nearest_neighbors(
    coords1: np.ndarray, coords2: np.ndarray, r: float
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute radius nearest neighbors between coords1 and coords2.
    The number of neighbors is different for each centeroid point, so return list data.

    Args:
        coords1: coordinates of centroid points (N, C)
        coords2: coordinates of all points (M, C)
        r: radius

    Returns:
        idxs: indices of neighbors within a radius
        square distances: square distance between pairwise points
    """

    # compute nearest neighbors.
    idxs, square_dists = k_nearest_neighbors(coords1, coords2, len(coords2))

    # get radius nearest neighbors masks.
    radius_masks = square_dists < r ** 2

    # get radius nearest neighbors according to masks
    radius_neighbor_indices = []
    radius_neighbor_square_dists = []
    for i, radius_mask in enumerate(radius_masks):
        radius_neighbor_indices.append(idxs[i, radius_mask])
        radius_neighbor_square_dists.append(square_dists[i, radius_mask])

    return radius_neighbor_indices, radius_neighbor_square_dists


def radius_and_k_nearest_neighbors(
    coords1: np.ndarray, coords2: np.ndarray, r: float, k: int = 32
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute radius and k nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (N, C)
        coords2: coordinates of all points (M, C)
        r: radius
        k: number of nearest neighbors

    Returns:
        idxs: indices of neighbors (N, k)
        square distances: square distance between pairwise points (N, k)
    """

    # compute kNN.
    idxs, square_dists = k_nearest_neighbors(coords1, coords2, k)

    # get radius nearest neighbors mask.
    radius_masks = square_dists < r ** 2

    # radius mask
    idxs[radius_masks == False] = len(coords2)
    square_dists[radius_masks == False] = -1

    return idxs, square_dists
