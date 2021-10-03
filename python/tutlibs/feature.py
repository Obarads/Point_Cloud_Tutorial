import numpy as np
import itertools # In order to get full connection of points, use itertools.combinations().

from .nns import k_nearest_neighbors, radius_nearest_neighbors
from .utils import grouping
from .utils import normalized_cross, angle

def ppf(xyz:np.ndarray, normals:np.ndarray) -> np.ndarray:
    idxs, square_dists = k_nearest_neighbors(xyz, xyz, k=2)
    F1s = np.sqrt(square_dists[:, 1])

    knn_normals = grouping(normals, idxs)
    unit_vector_0 = knn_normals[:, 0] / np.linalg.norm(knn_normals[:, 0], axis=1)[:, np.newaxis]
    unit_vector_1 = knn_normals[:, 1] / np.linalg.norm(knn_normals[:, 1], axis=1)[:, np.newaxis]

    knn_points = grouping(xyz, idxs)
    knn_vector = knn_points[:, 0] - knn_points[:, 1]
    unit_vector_2 = knn_vector / np.linalg.norm(knn_vector, axis=1)[:, np.newaxis]

    F2s = np.arccos(np.matmul(unit_vector_0[:, np.newaxis], unit_vector_2[:, :, np.newaxis])).squeeze(axis=(1,2))
    F3s = np.arccos(np.matmul(unit_vector_1[:, np.newaxis], unit_vector_2[:, :, np.newaxis])).squeeze(axis=(1,2))
    F4s = np.arccos(np.matmul(unit_vector_0[:, np.newaxis], unit_vector_1[:, :, np.newaxis])).squeeze(axis=(1,2))

    ppfs = np.stack([F1s, F2s, F3s, F4s], axis=-1)

    return ppfs

def pair_feature(xyz, normals) -> list:
    """Compute Point Feature Histograms (PFH).

    Args:
        xyz: xyz coords (N, 3)
        normals: normals (N, 3)
    Return:
        pair_feature: point feature
    """
    knn_idxs, _, rnn_masks = radius_nearest_neighbors(xyz, xyz, r=0.05, k=30)

    num_pf = len(knn_idxs)
    knn_xyzs = grouping(xyz, knn_idxs)
    knn_normals = grouping(normals, knn_idxs)

    pf_list = []
    for i in range(num_pf):
        # Get point set in radius
        rnn_mask = rnn_masks[i]
        rnn_xyz = knn_xyzs[i, rnn_mask]
        rnn_normal = knn_normals[i, rnn_mask]

        # Get pair points in radius, Pair is s and t.
        full_connection_idx = np.array(itertools.combinations(range(len(rnn_xyz)), 2))
        # Get point(xyz) and normal of s
        ps = rnn_xyz[full_connection_idx[:, 0]]
        ns = rnn_normal[full_connection_idx[:, 0]]
        # Get point(xyz) and normal of t
        pt = rnn_xyz[full_connection_idx[:, 1]]
        nt = rnn_normal[full_connection_idx[:, 1]]

        pp = ps - pt
        u = ns
        v = normalized_cross(pp, u)
        w = normalized_cross(u, v)

        phi = angle(pp, u)
        alpha = angle(v, nt)
        theta = np.arctan2(angle(w, nt), angle(u, nt))

        pf_list.append([alpha, phi, theta, pp])

    return pf_list

def pfh(xyz, normals):
    pf_list = pair_feature(xyz, normals)





