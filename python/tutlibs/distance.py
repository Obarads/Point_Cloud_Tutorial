import numpy as np
from . import operator

# TODO: duplicate tutlibs.operator
def square_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """Compute the square distances between coords1 and coords2.

    Note: Depending on the input value and the dtype, negative values may
    be mixed in the return value.

    Args:
        coords1: coordinates (N, C)
        coords2: coordinates (M, C)

    Returns:
        square distances:
            square distances between coords1 and coords2 (N, M)
    """
    res = -2 * np.matmul(coords1, coords2.T)
    res += np.sum(coords1 ** 2, axis=1, keepdims=True)
    res += np.sum(coords2 ** 2, axis=1, keepdims=True).T
    return res


def hausdorff_distance(points1: np.ndarray, points2: np.ndarray):
    if len(points2) < len(points1):
        temp = np.copy(points1)
        points1 = np.copy(points2)
        points2 = temp
    point_square_distances = square_distance(points1, points2)
    distance = np.max(np.min(point_square_distances, axis=1))

    return distance
