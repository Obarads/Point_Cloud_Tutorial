import numpy as np
import open3d as o3d

from .nns import k_nearest_neighbors, radius_nearest_neighbors
from .utils import grouping

def normal_estimation(coords:np.ndarray, k:int=10) -> np.ndarray:
    """Estimate normals each point with eigenvector of covariance matrix.
    This function use k nearest neighbor (kNN) to get covariance matrixs.

    Args:
        coords: coordinates of points, (N, 3)
        k: number of neighbor for kNN
    
    Returns
        normals: estimated normals 
    """
    # Get kNN. (TODO: add radius)
    idxs, _  = k_nearest_neighbors(coords, coords, k)
    knn_points = grouping(coords, idxs)

    # Get covariance matrix of each point
    knn_mean_points = np.mean(knn_points, axis=-2, keepdims=True)
    deviation = knn_points - knn_mean_points
    deviation_t = deviation.transpose(0, 2, 1)
    covariance_matrixs = np.matmul(deviation_t, deviation) / k # (N, 3, 3)

    # Get eigenvector and eigenvalue of each point
    w, v = np.linalg.eig(covariance_matrixs.transpose(0, 2, 1))
    # w, v = np.linalg.eig(covariance_matrixs)
    w_min_idxs = np.argmin(w, axis=1)

    # Get normal of each point
    normals = np.take_along_axis(v, w_min_idxs[:, np.newaxis, np.newaxis], axis=2)
    normals = normals.squeeze(2) # (N, normal_vector, 1) -> (N, normal_vector)
    # above code of normals is same code below:
    # normals = np.zeros((len(v), 3), dtype=np.float32) # (N, 3)
    # for i, idx in enumerate(w_min_idxs):
    #     normals[i] = v[i, :, idx]

    return normals

def normal_estimation_w_o3d(xyz:np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)
