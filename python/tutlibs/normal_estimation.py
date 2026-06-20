import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .nns import k_nearest_neighbors, radius_and_k_nearest_neighbors
from .operator import gather, square_distance, dot


# Note:
# (1) code is same code below:
# normals = np.zeros((len(v), 3), dtype=np.float32) # (N, 3)
# for i, idx in enumerate(w_min_idxs):
#     normals[i] = v[i, :, idx]
def _normals_from_neighbors(
    coords: np.ndarray, idxs: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Estimate normals from neighbor indices via PCA.

    Args:
        coords: coordinates of points used to gather neighbors, (M, 3)
        idxs: neighbor indices for each point, (N, K)
        valid_mask: True where a neighbor is valid, (N, K). Invalid neighbors
            (e.g. fewer than K neighbors within a radius) are ignored.

    Returns:
        normals: estimated normals (N, 3)
    """
    neighbor_points = gather(coords, idxs)  # (N, K, 3)

    # weights ignore invalid neighbors (count varies per point for radius NNS).
    weights = valid_mask[..., np.newaxis].astype(neighbor_points.dtype)  # (N, K, 1)
    num_neighbors = np.maximum(
        np.sum(weights, axis=1, keepdims=True), 1
    )  # (N, 1, 1)

    # Get covariance matrix of each point (only with valid neighbors).
    mean_points = np.sum(neighbor_points * weights, axis=1, keepdims=True) / num_neighbors
    deviation = (neighbor_points - mean_points) * weights
    deviation_t = deviation.transpose(0, 2, 1)
    covariance_matrixs = np.matmul(deviation_t, deviation) / num_neighbors.reshape(
        -1, 1, 1
    )  # (N, 3, 3)

    # Get eigenvector and eigenvalue for each point
    w, v = np.linalg.eig(covariance_matrixs.transpose(0, 2, 1))
    w_min_idxs = np.argmin(w, axis=1)

    # Get normal of each point (1)
    normals = np.take_along_axis(v, w_min_idxs[:, np.newaxis, np.newaxis], axis=2)
    normals = normals.squeeze(2)  # (N, normal_vector, 1) -> (N, normal_vector)

    return normals


def normal_estimation(
    coords: np.ndarray, k: int = 10, r: float = None
) -> np.ndarray:
    """Estimate normals each point with eigenvector of covariance matrix.
    This function uses nearest neighbor search to get covariance matrixs.

    Args:
        coords: coordinates of points, (N, 3)
        k: number of neighbor for nearest neighbor search
        r: radius for neighbor search. If None, use kNN. Otherwise, use up to
            k neighbors within the radius (radius and k nearest neighbors).

    Returns
        normals: estimated normals (N, 3)
    """
    # Get neighbor points.
    if r is None:
        idxs, _ = k_nearest_neighbors(coords, coords, k)
        valid_mask = np.ones_like(idxs, dtype=bool)
    else:
        idxs, _ = radius_and_k_nearest_neighbors(coords, coords, r, k)
        valid_mask = idxs != len(coords)  # invalid idx is set to len(coords).
        idxs = np.where(valid_mask, idxs, 0)  # make gather of invalid idx safe.

    return _normals_from_neighbors(coords, idxs, valid_mask)


def normal_estimation_v2(
    coords1: np.ndarray, coords2: np.ndarray, k: int = 10, r: float = None
) -> np.ndarray:
    """Estimate normals each point with eigenvector of covariance matrix.
    This function uses nearest neighbor search to get covariance matrixs.

    Args:
        coords1: coordinates of points, (N, 3)
        coords2: coordinates of points, (M, 3)
        k: number of neighbor for nearest neighbor search
        r: radius for neighbor search. If None, use kNN. Otherwise, use up to
            k neighbors within the radius (radius and k nearest neighbors).

    Returns
        normals: estimated normals (N, 3)
    """
    # Get neighbor points.
    if r is None:
        idxs, _ = k_nearest_neighbors(coords1, coords2, k)
        valid_mask = np.ones_like(idxs, dtype=bool)
    else:
        idxs, _ = radius_and_k_nearest_neighbors(coords1, coords2, r, k)
        valid_mask = idxs != len(coords2)  # invalid idx is set to len(coords2).
        idxs = np.where(valid_mask, idxs, 0)  # make gather of invalid idx safe.

    return _normals_from_neighbors(coords1, idxs, valid_mask)


def normal_orientation(coords: np.ndarray, normals: np.ndarray):
    """Normal orientation.

    Args:
        coords: (N, 3)
        normals: (N, 3)

    Returns:
        normals: (N, 3)
    """
    N, _ = normals.shape

    def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray):
        """Cosine similarity

        Args:
            vector1: (C)
            vector2: (C)

        Return:
            similarity: value range (-1 ~ 1)

        Note:
            C: number of vector dimension
        """
        return np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2)
        )

    def tree_recursion(
        _normals: np.ndarray,
        adjacency_matrix: np.ndarray,
        idx: int,
        rest_idxs: np.ndarray = None,
    ):
        """tree recursion for normal orientation

        Args:
            _normals: (N, 3)
            adjacency_matrix: adjacency matrix between points, (N, N)
            idx: first index for tree (first node)
            rest_idxs: rest node indexs for recursion processing, (N)

        Returns:
            oriented_normals: fixed normals(N, 3)
        """
        if rest_idxs is None:
            rest_idxs = np.full(N, fill_value=True, dtype=np.bool8)
        mask_row = adjacency_matrix[idx] > 0
        mask_col = adjacency_matrix[:, idx] > 0
        mask = mask_row + mask_col
        connection = np.arange(N)[rest_idxs & mask]

        for c_idx in connection:
            rest_idxs[c_idx] = False
            parent = _normals[idx]
            child = _normals[c_idx]

            cos_sim = cosine_similarity(parent, child)
            cos_sim_minus = cosine_similarity(parent, -child)

            if cos_sim_minus > cos_sim:
                _normals[c_idx] = child * -1

            _normals = tree_recursion(_normals, adjacency_matrix, c_idx, rest_idxs)

        return _normals

    _normals = np.copy(normals)
    weighted_adjacency_matrix = square_distance(coords, coords)
    weighted_adjacency_matrix[np.identity(N, dtype=np.bool8)] = 0.0
    tcsr = minimum_spanning_tree(weighted_adjacency_matrix)
    tree_weighted_adjacency_matrix = tcsr.toarray()

    _normals = tree_recursion(_normals, tree_weighted_adjacency_matrix, 0)

    return _normals


def normal_orientation_with_viewpoint(
    coords: np.ndarray, normals: np.ndarray, viewpoint: np.ndarray
):
    """Normal orientation with viewpoint

    Args:
        coords: xyz coordinates of points, (N, 3)
        normals: normals of points, (N, 3)
        viewpoint: view point, (3)

    Returns:
        oriented_normals: fixed normals(N, 3)
    """
    _normals = np.copy(normals)
    direction_dot = dot(_normals, viewpoint[np.newaxis, :] - coords)
    orientation_mask = direction_dot > 0
    _normals[orientation_mask] *= -1

    return _normals
