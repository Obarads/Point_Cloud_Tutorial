"""operator"""

import numpy as np


def dot(a, b, keepdims=False) -> np.ndarray:
    """Dot product for a and b vectors
    """
    res: np.ndarray = np.matmul(a[:, np.newaxis], b[:, :, np.newaxis])
    if not keepdims:
        res = res.squeeze()
    return res


def cross(a, b) -> np.ndarray:
    """Cross product for a and b vectors
    """
    res: np.ndarray = np.cross(a, b)
    return res


def normalize(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors.

    Args:
        vectors: (N, C)

    Return:
        vectors: (N, C)
    """
    return vectors / np.linalg.norm(vectors, ord=2, axis=1)[:, np.newaxis]


def angle(a: np.ndarray, b: np.ndarray, norm_skip: bool = False) -> np.ndarray:
    """Angle between a and b vectors

    Args:
        a: vectors a (N, C)
        b: vectors b (N, C)
        norm_skip: whether this function skip normilizetion for a and b.

    Note:
        If a and b was normilized, you should set `norm_skip=True`.
        This function calculations will be more accurate and faster.
    """
    if not norm_skip:
        a = normalize(a)
        b = normalize(b)
    res = np.arccos(dot(a, b))
    return res


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
    res = -2*np.matmul(coords1, coords2.T)
    res += np.sum(coords1**2, axis=1, keepdims=True)
    res += np.sum(coords2**2, axis=1, keepdims=True).T
    return res


def gather(points: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Construct points according to indices.
    This function is similar to `gather` (EX. tf.gather).

    Args:
        points: points (N, C)
        idx: indices (N) or (N, k)

    Returns:
        new points: points according to indices

    To use:
    >>> import numpy as np
    >>> points = np.arange(24).reshape(8,3)
    >>> points
    array([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23]])
    >>> idx = np.array([2, 4, 1, 1, 0, 3, 7])
    >>> grouping2(points, idx)
    array([[ 6,  7,  8],
        [12, 13, 14],
        [ 3,  4,  5],
        [ 3,  4,  5],
        [ 0,  1,  2],
        [ 9, 10, 11],
        [21, 22, 23]])

    >>> idx = np.array([[2, 3, 5],[1, 0, 0]])
    >>> N, k = idx.shape # k (such as number of k nearest neighbors)
    >>> grouping2(points, idx)
    array([[[ 6,  7,  8],
            [ 9, 10, 11],
            [15, 16, 17]],

        [[ 3,  4,  5],
            [ 0,  1,  2],
            [ 0,  1,  2]]])
    """

    # get original shape
    idx_shape = idx.shape
    N, C = points.shape

    # reshape to a row
    row_idx = idx.reshape(-1)

    # get grouped points
    row_points = points[row_idx]
    new_points = row_points.reshape(*idx_shape, C)

    return new_points
