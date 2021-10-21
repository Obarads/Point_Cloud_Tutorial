from typing import List
import numpy as np

from .nns import k_nearest_neighbors
from .utils import grouping, square_distance
from .transformation import TransformationMatrix as tm

def brute_force_matching(coords1, coords2, k=1) -> np.ndarray:
    idx, _ = k_nearest_neighbors(coords1, coords2, k)
    return idx

def estimate_transformation_with_svd(source:np.ndarray,
                                     target:np.ndarray) -> np.ndarray:
    """Estimate transformation with Singular Value Decomposition (SVD).

    Args:
        source: (N, 3)
        target: (N, 3)
    
    Returns:
        R: rotation matrix, (3, 3)
        T: translation vector, (3)
    """

    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    source = source - centroid_source
    target = target - centroid_target

    H = np.matmul(source.T, target)
    u, _, vh = np.linalg.svd(H)
    R = np.matmul(vh.T, u.T)
    T = centroid_target - np.matmul(R, centroid_source)

    return R, T

def estimate_transformation_with_svd_and_corr_set(
        source:np.ndarray, target:np.ndarray, corr_set:np.ndarray
        ) -> np.ndarray:
    """Estimate transformation with Singular Value Decomposition (SVD).

    Args:
        source: coordinates of source points, (N, 3)
        target: coordinates of target points, (M, 3)
        corr_set: correspondence index between source and target (L, 2) 

    Returns:
        R: rotation matrix, (3, 3)
        T: translation vector, (3)
    """

    source = source[corr_set[:, 0]]
    target = target[corr_set[:, 1]]

    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    source = source - centroid_source
    target = target - centroid_target

    e = np.matmul(source.T, target)
    u, s, vh = np.linalg.svd(e)
    R = np.matmul(vh.T, u.T)
    T = np.matmul(-R, centroid_source) + centroid_target

    return R, T

def icp(source:np.ndarray, target:np.ndarray, iteration:int,
        threshold:float=0.0000001, init_trans_mat:np.ndarray=None) -> np.ndarray:
    """Iterative Closest Point

    Args:
        source: coordinates of source points, (N, 3)
        target: coordinates of target points, (M, 3)
        iteration: icp iteration
        threshold: convergence threshold
        init_trans_mat: initialinitial transformation matrix, (4, 4)

    Return:
        trans_mat: (4, 4)
    """

    if init_trans_mat is None:
        init_trans_mat = np.identity(4)

    target = np.copy(target)
    source = np.copy(source)
    trans_mat = np.copy(init_trans_mat) # transformation matrix
    # new_trans_mat = np.copy(init_trans_mat) # previus transformation matrix

    for _ in range(iteration):
        trans_source = tm.transformation_Nx3_with_4x4(source, trans_mat)
        idxs, _ = k_nearest_neighbors(trans_source, target, k=2)
        correspondence_target = grouping(target, idxs)[:, 1]

        R, T = estimate_transformation_with_svd(trans_source, correspondence_target)
        new_trans_source = tm.transformation_Nx3_with_3x3_3(trans_source, R, T)

        R_mat = tm.rotation_3x3_to_4x4(R)
        T_mat = tm.translation_3_to_4x4(T)
        trans_mat = tm.composite_4x4([T_mat, R_mat, trans_mat])

        if np.sum(np.abs(new_trans_source - trans_source)) < threshold:
            break

    return trans_mat

def correspondence_rejection_with_distance(source_xyz:np.ndarray, target_xyz:np.ndarray,
                                           correspondence_indices:np.ndarray,
                                           threshold:float):
    """Correspondence rejection with distance threshold between source xyz and target xyz,
    Corresponding points whose distance between source and target exceeds the threshold will be deleted by this function.

    Args:
        source_xyz: coordinates of source points, (N, 3)
        target_xyz: coordinates of target points, (M, 3)
        correspondence_indices: target xyz indices corresponding to source xyz, (N, 3)
        threshold: distance threshold
        
    Return:
        new_source_indices: new source indices, (L, 3)
    """
    
    correspondence_xyz = target_xyz[correspondence_indices]
    two_point_distance = source_xyz - correspondence_xyz
    new_source_indices = np.arange(source_xyz.shape[0])[two_point_distance > threshold]
    return new_source_indices

def correspondence_ransac(source_xyz:np.ndarray, target_xyz:np.ndarray,
                          source_feature:np.ndarray, target_feature:np.ndarray,
                          corr_set:np.ndarray, ransac_n:int,
                          iter:int, threshold:float, 
                          checkers:list=[]):
    """Correspondence rejection with RANSAC
    
    Args:
        source_xyz: coordinates of source points (N, 3)
        target_xyz: coordinates of target points (M, 3)
        source_feature: features of source points (N, C)
        target_feature: features of target points (M, C)
        corr_set: correspondence index between source and target (L, 2)
        ransac_n: number of random sampling points for RANSAC, ransac_n <= L
        iter: number of iteration 
        threshold: RANSAC threshold
        checkers: correspondence rejection functions

    Returns:
        ransac_corr: new correspondence indices, (J, 3)
        fitness: number of inlier / ransac_corr
        inlier_rmse: RMSE of distances between new 2 point correspondence
    """

    # Start RANSAC.
    for i in range(iter):
        # Define random sampling for RANSAC
        corr_random_idxs = np.random.default_rng().choice(corr_set.shape[0], size=ransac_n, replace=False)
        ransac_corr = corr_set[corr_random_idxs]

        # Estimate transformation with ransac correspondetion
        R, T = estimate_transformation_with_svd_and_corr_set(source_xyz, target_xyz, ransac_corr)
        trans_source_xyz = tm.transformation_Nx3_with_3x3_3(source_xyz, R, T)

        # Use correspondence rejection.
        for checker in checkers:
            checked_idxs = checker(trans_source_xyz, target_xyz, ransac_corr)
            ransac_corr = ransac_corr[checked_idxs]
            
        # RANSAC evaluation
        square_threshold = threshold ** 2
        square_corr_dists = square_distance(trans_source_xyz, target_xyz)[ransac_corr[:, 0], ransac_corr[:, 1]]
        inliers = np.arange(square_corr_dists.shape[0])[square_corr_dists < square_threshold]
        num_inlier = inliers.shape[0]
        error = np.sum(square_corr_dists[square_corr_dists < square_threshold])

        # evaluation value check
        if num_inlier == 0:
            fitness = 0.0
            inlier_rmse = 0.0
        else:
            fitness = num_inlier / ransac_corr.shape[0]
            inlier_rmse = error / num_inlier

    return ransac_corr, fitness, inlier_rmse



