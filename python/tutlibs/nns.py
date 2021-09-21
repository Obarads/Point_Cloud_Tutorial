import numpy as np
from .utils import square_distance

def k_nearest_neighbors(coords1:np.ndarray, coords2:np.ndarray, k:int):
        """
        Compute k nearest neighbors between coords1 and coords2.

        Args:
            coords1: coordinates of center points (N, C)
            coords2: coordinates of all points (M, C)
            k: number of nearest neighbors

        Returns:
            idxs: indices of k nearest neighbors (N, k)
            square distances: square distance for kNN (N, k)
        """

        # batch proc.
        point_pairwise_distances = square_distance(coords1, coords2)
        idxs = np.argsort(point_pairwise_distances, axis=-1)[:, :k]
        square_dists = np.take_along_axis(point_pairwise_distances, idxs, axis=-1)

        return idxs, square_dists


def radius_nearest_neighbors(coords1:np.ndarray, coords2:np.ndarray, r:float, k:int=32):
    """
    Compute radius nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of center points (N, C)
        coords2: coordinates of all points (M, C)
        r: radius
        k: number of nearest neighbors

    Returns:
        idxs: indices of k nearest neighbors (N, k)
        square distances: square distance for kNN (N, k)
        mask : radius mask for idxs an distance (N, k)
    """

    idxs, square_dists = k_nearest_neighbors(coords1, coords2, k)
    radius_mask = square_dists < r

    return idxs, square_dists, radius_mask