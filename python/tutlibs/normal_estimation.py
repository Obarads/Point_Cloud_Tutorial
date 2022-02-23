import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from .nns import k_nearest_neighbors
from .operator import gather, square_distance, dot

# Note:
# (1) code is same code below:
# normals = np.zeros((len(v), 3), dtype=np.float32) # (N, 3)
# for i, idx in enumerate(w_min_idxs):
#     normals[i] = v[i, :, idx]
def normal_estimation(coords: np.ndarray, k: int = 10) -> np.ndarray:
    """Estimate normals each point with eigenvector of covariance matrix.
    This function use k nearest neighbor (kNN) to get covariance matrixs.

    Args:
        coords: coordinates of points, (N, 3)
        k: number of neighbor for kNN

    Returns
        normals: estimated normals (N, 3)
    """
    # Get neighbor points. (TODO: add radius)
    idxs, _ = k_nearest_neighbors(coords, coords, k)
    knn_points = gather(coords, idxs)

    # Get covariance matrix of each point.
    knn_mean_points = np.mean(knn_points, axis=-2, keepdims=True)
    deviation = knn_points - knn_mean_points
    deviation_t = deviation.transpose(0, 2, 1)
    covariance_matrixs = np.matmul(deviation_t, deviation) / k  # (N, 3, 3)

    # Get eigenvector and eigenvalue of each point
    w, v = np.linalg.eig(covariance_matrixs.transpose(0, 2, 1))
    # w, v = np.linalg.eig(covariance_matrixs)
    w_min_idxs = np.argmin(w, axis=1)

    # Get normal of each point (1)
    normals = np.take_along_axis(v, w_min_idxs[:, np.newaxis, np.newaxis], axis=2)
    normals = normals.squeeze(2)  # (N, normal_vector, 1) -> (N, normal_vector)

    return normals


def normal_estimation_v2(
    coords1: np.ndarray, coords2: np.ndarray, k: int = 10
) -> np.ndarray:
    """Estimate normals each point with eigenvector of covariance matrix.
    This function use k nearest neighbor (kNN) to get covariance matrixs.

    Args:
        coords1: coordinates of points, (N, 3)
        coords2: coordinates of points, (M, 3)
        k: number of neighbor for kNN

    Returns
        normals: estimated normals (N, 3)
    """
    # Get neighbor points. (TODO: add radius)
    idxs, _ = k_nearest_neighbors(coords1, coords2, k)
    knn_points = gather(coords1, idxs)

    # Get covariance matrix of each point.
    knn_mean_points = np.mean(knn_points, axis=-2, keepdims=True)
    deviation = knn_points - knn_mean_points
    deviation_t = deviation.transpose(0, 2, 1)
    covariance_matrixs = np.matmul(deviation_t, deviation) / k  # (N, 3, 3)

    # Get eigenvector and eigenvalue of each point
    w, v = np.linalg.eig(covariance_matrixs.transpose(0, 2, 1))
    # w, v = np.linalg.eig(covariance_matrixs)
    w_min_idxs = np.argmin(w, axis=1)

    # Get normal of each point (1)
    normals = np.take_along_axis(v, w_min_idxs[:, np.newaxis, np.newaxis], axis=2)
    normals = normals.squeeze(2)  # (N, normal_vector, 1) -> (N, normal_vector)

    return normals


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
