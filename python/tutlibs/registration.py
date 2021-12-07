import numpy as np

from .transformation import TransformationMatrix as tm
from .operator import square_distance
from .nns import k_nearest_neighbors

def brute_force_matching(coords1:np.ndarray, coords2:np.ndarray) -> np.ndarray:
    """Brute force matching
    
    Args:
        coords1: point cloud feature (N, C)
        coords2: point cloud feature (M, C)

    Return
        corr_set: correspondence set, [..., [coord1_idx, coord2_idx], ...] (N, 2)
    """
    coords2_idx, _ = k_nearest_neighbors(coords1, coords2, 1)
    coords1_idx = np.arange(coords1.shape[0]).reshape(-1, 1)
    corr_set = np.hstack([coords1_idx, coords2_idx])
    return corr_set

class CorrespondenceRejection:
    @staticmethod
    def distance_between_points(source_xyz:np.ndarray, target_xyz:np.ndarray,
                                corr_set:np.ndarray,
                                threshold:float=0.05):
        """Correspondence rejection with distance threshold between source xyz and target xyz,
        Corresponding points whose distance between source and target exceeds the threshold will be deleted by this function.

        Args:
            source_xyz: coordinates of source points, (N, 3)
            target_xyz: coordinates of target points, (M, 3)
            corr_set: target xyz indices corresponding to source xyz, (N, 3)
            threshold: distance threshold

        Return:
            new_source_indices: new source indices, (L, 3)
        """
        correspondence_xyz = target_xyz[corr_set]
        two_point_distance = source_xyz - correspondence_xyz
        new_source_indices = np.arange(source_xyz.shape[0])[two_point_distance > threshold]
        return new_source_indices

    @staticmethod
    def ransac(source_xyz:np.ndarray, target_xyz:np.ndarray,
               corr_set:np.ndarray, ransac_n:int, iteration:int, 
               threshold:float):
        """Correspondence rejection with RANSAC.
        
        Args:
            source_xyz: coordinates of source points (N, 3)
            target_xyz: coordinates of target points (M, 3)
            corr_set: correspondence index between source and target (L, 2)
            ransac_n: number of random sampling points for RANSAC, ransac_n <= L
            iter: number of iteration 
            threshold: RANSAC threshold

        Returns:
            ransac_corr: new correspondence indices, (J, 3)
            num_inliner: number of inlier
            inlier_rmse: RMSE of distances between new 2 point correspondence
        """

        best_ransac_corr = corr_set
        best_inlier_ratio = 0
        best_inlier_rmse = 0

        # Start RANSAC.
        for _ in range(iteration):
            # define random sampling.
            corr_random_idxs = np.random.default_rng().choice(corr_set.shape[0],
                                                              size=ransac_n,
                                                              replace=False)
            ransac_corr = corr_set[corr_random_idxs]

            # estimate transformation with ransac correspondetion.
            corr_trans_mat = estimate_transformation(source_xyz,
                                                     target_xyz,
                                                     ransac_corr)
            trans_source_xyz = tm.transformation(source_xyz, corr_trans_mat)

            # get inlier information.
            corr_square_dists = square_distance(trans_source_xyz, target_xyz)[corr_set[:, 0], corr_set[:, 1]]
            inlier_mask = corr_square_dists < threshold ** 2
            inlier_corr_set = corr_set[inlier_mask]
            inlier_square_dists = corr_square_dists[inlier_mask]
            num_inlier = len(inlier_corr_set)

            # evaluation
            if num_inlier > 0:
                inlier_rmse = np.mean(inlier_square_dists)
                inlier_ratio = num_inlier / len(corr_set)
                if inlier_ratio > best_inlier_ratio or (inlier_ratio == best_inlier_ratio and inlier_rmse < best_inlier_rmse):
                    best_ransac_corr = inlier_corr_set
                    best_inlier_ratio = inlier_ratio
                    best_inlier_rmse = inlier_rmse

        return best_ransac_corr, best_inlier_ratio, best_inlier_rmse

def estimate_transformation(
        source:np.ndarray, target:np.ndarray, corr_set:np.ndarray
        ) -> np.ndarray:
    """Estimate transformation with Singular Value Decomposition (SVD).

    Args:
        source: coordinates of source points, (N, 3)
        target: coordinates of target points, (M, 3)
        corr_set: correspondence index between source and target (L, 2)

    Returns:
        rotation_3x3: rotation matrix, (3, 3)
        translation_3: translation vector, (3)
    """

    source = source[corr_set[:, 0]]
    target = target[corr_set[:, 1]]

    centroid_source = np.mean(source, axis=0)
    centroid_target = np.mean(target, axis=0)

    source = source - centroid_source
    target = target - centroid_target

    correlation_mat = np.matmul(source.T, target)
    u, _, vh = np.linalg.svd(correlation_mat)
    rotation_3x3 = np.matmul(vh.T, u.T)
    translation_3 = centroid_target - np.matmul(rotation_3x3, centroid_source)

    trans_mat = tm.composite([
        tm.from_translation(translation_3),
        tm.from_rotation(rotation_3x3)
    ])
    
    return trans_mat

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
    trans_mat = init_trans_mat.copy() # transformation matrix

    target = target.copy()
    source = source.copy()

    for _ in range(iteration):
        # transform source xyz with trans_mat
        trans_source = tm.transformation(source, trans_mat)

        # find correspondece set between target and trans_source
        corr_set = brute_force_matching(trans_source, target)

        # compute transformation matrix between trans_source and target with corr_set
        corr_trans_mat = estimate_transformation(trans_source, target, corr_set)
        
        # transform trans_source with corr_trans_mat
        new_trans_source = tm.transformation(trans_source, corr_trans_mat)

        # update trans_mat
        trans_mat = tm.composite([corr_trans_mat, trans_mat])

        if np.sum(np.abs(new_trans_source - trans_source)) < threshold:
            break

    return trans_mat

def feature_ransac(source_xyz:np.ndarray, target_xyz:np.ndarray,
                   source_feature:np.ndarray, target_feature:np.ndarray,
                   ransac_n:int, iteration:int, threshold:float=0.0000001,
                   init_trans_mat:np.ndarray=None) -> np.ndarray:
    """

    Args:
        source_xyz: coordinates of source points, (N, 3)
        target_xyz: coordinates of target points, (M, 3)
        source_features: features of source points, (N, C)
        target_features: features of target points, (M, C)
        ransac_n: number of samples
        iteration: icp iteration
        threshold: convergence threshold
        init_trans_mat: initialinitial transformation matrix, (4, 4)

    Return:
        trans_mat: (4, 4)
    """

    # set transformation matrix
    if init_trans_mat is None:
        init_trans_mat = np.identity(4)
    trans_mat = init_trans_mat.copy()

    # data copy
    target_xyz = target_xyz.copy()
    source_xyz = source_xyz.copy()
    target_feature = target_feature.copy()
    source_feature = source_feature.copy()

    # correspondence baed features
    corr_set_feature = brute_force_matching(source_feature, target_feature)

    # source xyz with corrent transformation
    trans_source_xyz = tm.transformation(source_xyz, trans_mat)

    # correspondence rejection for corr_set_feature
    ransac_corr_set, _, _ = CorrespondenceRejection.ransac(trans_source_xyz,
                                                           target_xyz,
                                                           corr_set_feature,
                                                           ransac_n, iteration,
                                                           threshold)

    # transformation estimation with ransac_rcorr_set
    corr_trans_mat = estimate_transformation(trans_source_xyz,
                                             target_xyz,
                                             ransac_corr_set)

    # create new transformation matrix
    trans_mat = tm.composite([corr_trans_mat, trans_mat])

    return trans_mat
