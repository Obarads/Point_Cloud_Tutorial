import numpy as np
from .nns import k_nearest_neighbors
from .utils import grouping

def brute_force_matching(coords1, coords2, k=1) -> np.ndarray:
    idx, _ = k_nearest_neighbors(coords1, coords2, k)
    return idx

def ransac(source_xyz:np.ndarray, target_xyz:np.ndarray, init_trans_matrix:np.ndarray, iter:int):
    """(TODO) Correspondence rejection with RANSAC
    
    Args:
        source_xyz: coordinates of source points (N, 3)
        target_xyz: coordinates of target points (M, 3)
        init_trans_matrix: initial transformation matrix, (3, 3).
        iter: Iteration 
    """
    return
