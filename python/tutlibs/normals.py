import numpy as np
import open3d as o3d

from nns import k_nearest_neighbors
from utils import grouping

def normal_estimation(coords:np.ndarray) -> np.ndarray:
    # Get kNN.
    k = 10
    idxs, dist = k_nearest_neighbors(coords, coords, k)
    knn_points = grouping(coords, idxs)

    # Get covariance matrix of each point
    knn_mean_points = np.mean(knn_points, axis=-2, keepdims=True)
    deviation = knn_points - knn_mean_points
    deviation_t = deviation.transpose(0, 2, 1)
    covariance_matrixs = np.matmul(deviation_t, deviation) / k # (N, 3, 3)

    # Get eigenvector and eigenvalue of each point
    w, v = np.linalg.eig(covariance_matrixs.transpose(0, 2, 1))
    idxs = np.argmin(w, axis=1)

    # Get normal of each point
    # normals = np.take_along_axis(v, idxs[:, np.newaxis, np.newaxis], axis=1)
    # normals = normals.squeeze(1)
    # above code of normals is same code below:
    normals = np.zeros((len(v), 3), dtype=np.float32) # (N, 3)
    for i, idx in enumerate(idxs):
        normals[i] = v[i, idx]

    return normals

def noraml_estimation_w_o3d(xyz:np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return np.asarray(pcd.normals)
