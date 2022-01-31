import numpy as np
import itertools

from .nns import radius_and_k_nearest_neighbors, k_nearest_neighbors
from .operator import gather, dot, cross, normalize, angle

def pair_feature(xyz:np.ndarray, normals:np.ndarray, pair_idxs:np.ndarray):
    """Compute pair features.

    Args:
        xyz: xyz coords (M, 3)
        normals: normals (M, 3)
        pair_idxs: point pair indices (L, 2)
    Return:
        phi: cosine feature, value range is -1~1 (L)
        alpha: cosine feature, value range is -1~1 (L)
        theta: angle feature, value range is -pi/2~pi/2 (L)
        dists: distance feature (L)
    """

    # Get xyz and normal of points
    point_1_indices = pair_idxs[:, 0]
    point_2_indices = pair_idxs[:, 1]
    p1 = xyz[point_1_indices]
    n1 = normals[point_1_indices]
    p2 = xyz[point_2_indices]
    n2 = normals[point_2_indices]

    # Get vectors (pp) and distances (dists) between pt and ps
    pp = p2 - p1
    dists = np.linalg.norm(pp, ord=2, axis=1)

    # Get a mask to decide the source and target points.
    p1pp_angle = dot(n1, pp) / dists
    p2pp_angle = dot(n2, pp) / dists
    mask = np.arccos(np.fabs(p1pp_angle)) > np.arccos(np.fabs(p2pp_angle))

    # Decide source and target points.
    phi = p1pp_angle.copy()
    u = n1.copy() # u = ns
    nt = n2.copy()
    u[mask] = n2[mask].copy()
    nt[mask] = n1[mask].copy()
    pp[mask] *= -1
    phi[mask] = -1 * p2pp_angle[mask]

    # Compute v and w
    v = normalize(cross(pp, u))
    w = cross(u, v)

    # Get alpha and theta
    alpha = dot(v, nt)
    theta = np.arctan2(dot(w, nt), dot(u, nt))

    return phi, alpha, theta, dists

def fminmax(arr:np.ndarray, min_value:float, max_value:float):
    """Limits array values to a min <= arr <= max range.
    Args:
        arr: array
        min_value: minimum value
        max_value: maxinum value
    """
    arr = np.fmax(min_value, arr)
    arr = np.fmin(max_value, arr)
    return arr

class PairPointFeature:
    @staticmethod
    def compute(xyz:np.ndarray, normals:np.ndarray) -> np.ndarray:
        idxs, _ = k_nearest_neighbors(xyz, xyz, k=2)

        knn_normals = gather(normals, idxs)
        knn_xyz = gather(xyz, idxs)

        mm = knn_xyz[:, 1] - knn_xyz[:, 0]
        n1 = knn_normals[:, 0]
        n2 = knn_normals[:, 1]

        F1s = np.linalg.norm(mm, ord=2, axis=1)
        F2s = angle(n1, mm)
        F3s = angle(mm, n2)
        F4s = angle(n2, n1, norm_skip=True)

        ppfs = np.stack([F1s, F2s, F3s, F4s], axis=-1)
        return ppfs

class PointFeatureHistograms:
    @staticmethod
    def compute(xyz:np.ndarray, normals:np.ndarray, radius=0.03, div:int=5, normalization:bool=True) -> np.ndarray:
        """Compute Point Feature Histograms.
        Note: does not include the distance feature (reason -> https://pcl.readthedocs.io/projects/tutorials/en/latest/pfh_estimation.html#pfh-estimation)

        Args:
            xyz: xyz coords (N, 3)
            normals: normals (N, 3)
            radius: radius for nearest neighbor search
            div: number of subdivisions of value range for each feature.
            normalization: normalize each point histgram.
        Return:
            point feature histograms: (N, div**3)
        """
        knn_idxs, _, rnn_masks = radius_and_k_nearest_neighbors(xyz, xyz, r=radius, k=xyz.shape[0]) # shpae: (N, N), (N, N)

        # Get nearest neihobrs (get radius neihbors).
        num_pf = len(knn_idxs) # num_pf = N = xyz.shape[0]
        knn_xyzs = gather(xyz, knn_idxs) # shape: (N, N, 3)
        knn_normals = gather(normals, knn_idxs) # shape: (N, N, 3)

        # define PFH list
        pfh_list = np.zeros((num_pf, div** 3))

        for i in range(num_pf):
            # Get point set in radius (R=number of points in radius, R is different for each i.)
            rnn_mask = rnn_masks[i] # shape: (N)
            rnn_xyz = knn_xyzs[i, rnn_mask] # shape: (R, 3)
            rnn_normal = knn_normals[i, rnn_mask] # shape: (R, 3)

            if len(rnn_xyz) >= 2:
                pair_idxs = np.array(list(itertools.combinations(range(len(rnn_xyz)), 2))) # shape: (T, 2), T=Number of combinations
                phi, alpha,  theta, _ = pair_feature(rnn_xyz, rnn_normal, pair_idxs) # shape: (T), (T), (T), (T)

                theta_bin_idx = fminmax(np.floor((div * (theta + np.pi)) / (2.0 * np.pi)), 0, div - 1)
                alpha_bin_idx = fminmax(np.floor(div * ((alpha + 1.0) * 0.5)), 0, div - 1)
                phi_bin_idx = fminmax(np.floor(div * ((phi + 1.0) * 0.5)), 0, div - 1)

                histgram_idx = (phi_bin_idx * (div ** 2) + alpha_bin_idx * (div) + theta_bin_idx).astype(np.int32)
                histgram = np.bincount(histgram_idx, minlength=div ** 3).astype(np.float32)
                if normalization:
                    histgram /= (np.sum(rnn_mask) * (np.sum(rnn_mask) - 1) / 2) # ex: np.sum(histgram) = 1
            else:
                histgram = np.zeros(div ** 3)

            pfh_list[i] = histgram

        return pfh_list

class SimplifiedPointFeatureHistogram:
    @staticmethod
    def compute(xyz:np.ndarray, normals:np.ndarray, radius=0.03, div:int=5, normalization:bool=True) -> np.ndarray:
        """Compute Simplified Point Feature Histograms.

        Args:
            xyz: xyz coords (N, 3)
            normals: normals (N, 3)
            radius: radius for nearest neighbor search
            div: number of subdivisions of value range for each feature
            normalization: normalize each point histgram.
        Return:
            fast point feature histograms: (N, div**3)
        """
        N, _ = xyz.shape

        # radius neighobrs
        _, _, rnn_masks = radius_and_k_nearest_neighbors(xyz, xyz, r=radius, k=xyz.shape[0])

        # get pair idxs for radius neighbors
        pair_mask = np.triu(rnn_masks, k=-1) # shape: (N, N)
        all_pair_idxs = np.stack([np.tile(np.arange(N)[:, np.newaxis], (1, N)),
                                  np.tile(np.arange(N)[:, np.newaxis], (1, N)).T
                                  ], axis=-1) # shape: (N, N, 2)
        pair_idxs = all_pair_idxs[pair_mask] # shape: (S, 2)

        # define SPFH list.
        spfh_list = np.zeros(N, div**3)

        # compute pair features
        if len(pair_idxs) >= 1:
            phi, alpha, theta, _ = pair_feature(xyz, normals, pair_idxs)
            pair_features = np.stack((phi, alpha, theta), axis=-1)
            nn_arr = np.zeros((N, N, 3), dtype=pair_features.dtype)
            nn_arr[pair_idxs[:, 0], pair_idxs[:, 1]] = pair_features
            nn_arr[pair_idxs[:, 1], pair_idxs[:, 0]] = pair_features

            for i in range(N):
                # Get point set in radius (R=number of points in radius, R is different for each i.)
                rnn_mask = rnn_masks[i] # shape: (N)
                nn = nn_arr[i][rnn_mask]
                phi, alpha, theta = nn[:, 0], nn[:, 1], nn[:, 2]

                theta_bin_idx = fminmax(np.floor((div * (theta + np.pi)) / (2.0 * np.pi)), 0, div - 1)
                alpha_bin_idx = fminmax(np.floor(div * ((alpha + 1.0) * 0.5)), 0, div - 1)
                phi_bin_idx = fminmax(np.floor(div * ((phi + 1.0) * 0.5)), 0, div - 1)

                histgram_idx = (phi_bin_idx * (div ** 2) + alpha_bin_idx * (div) + theta_bin_idx).astype(np.int32)
                histgram = np.bincount(histgram_idx, minlength=div ** 3).astype(np.float32)
                if normalization:
                    histgram /= (np.sum(rnn_mask) * (np.sum(rnn_mask) - 1) / 2) # ex: np.sum(histgram) = 1
                spfh_list[i] = histgram

        return spfh_list

class FastPointFeatureHistograms:
    @staticmethod
    def compute(xyz:np.ndarray, normals:np.ndarray, radius=0.03, div:int=5) -> np.ndarray:
        """Compute Fast Point Feature Histograms.

        Args:
            xyz: xyz coords (N, 3)
            normals: normals (N, 3)
            radius: radius for nearest neighbor search
            div: number of subdivisions of value range for each feature
        Return:
            fast point feature histograms: (N, div**3)
        """
        spfh = SimplifiedPointFeatureHistogram.compute(xyz, normals, radius, div) # shape: (N, div**3)

        knn_idxs, knn_dists, rnn_masks = radius_and_k_nearest_neighbors(xyz, xyz, r=radius, k=xyz.shape[0]) # shpae: (N, N), (N, N), (N, N)

        # Get nearest neihobrs (get radius neihbors).
        num_pf = len(knn_idxs) # num_pf = N = xyz.shape[0]
        knn_spfh = gather(spfh, knn_idxs) # shape: (N, N. div**3)

        # define FPFH list.
        fpfh_list = np.zeros(num_pf, div**3)

        for i in range(num_pf):
            # Get point set in radius (R=number of points in radius, R is different for each i.)
            rnn_mask = rnn_masks[i] # shape: (N)
            rnn_spfh = knn_spfh[i, rnn_mask] # shape: (R, div**3)
            rnn_dist = knn_dists[i, rnn_mask]

            if len(rnn_spfh) >= 2:
                histgram = rnn_spfh[0] + np.mean(rnn_dist[1:] * rnn_spfh[1:], axis=0)
            else:
                histgram = rnn_spfh[0]

            fpfh_list[i] = histgram

        return fpfh_list
