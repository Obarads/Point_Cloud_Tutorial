import numpy as np
import pandas as pd

def furthest_point_sampling(coords:np.ndarray, num_sample:int) -> np.ndarray:
    """Furthest point sampling

    Args:
        coords: coords (N, C)
        num_sample: number of sammple
    
    Returns:
        indices: sample indices (num_sample)
    """
    N, C = coords.shape
    # Get min square dists between samples and all points
    min_square_dists = np.full(N, fill_value=2**16 - 1, dtype=np.float32)
    # Get sample indices (Return value)
    sample_indices = []
    
    # Get first index
    furthest_index = np.random.randint(0, N)
    for _ in range(num_sample - 1):
        sample_indices.append(furthest_index)
        furthest_point = coords[furthest_index]
        pairwise_dists = np.sum((coords - furthest_point[np.newaxis, :])**2, axis=1)
        min_update_mask = pairwise_dists < min_square_dists
        min_square_dists[min_update_mask] = pairwise_dists[min_update_mask]
        furthest_index = np.argmax(min_square_dists)

    return np.array(sample_indices, dtype=np.int32)

def voxel_grid_sampling(coords:np.ndarray, voxel_size:float) -> np.ndarray:
    """Voxel grid sampling

    Args:
        coords: coords (N, C)
        voxel_size: voxel grid size
    
    Returns:
        samples: sample coords (M, C)
    """
    N, C = coords.shape
    indices_float = coords / voxel_size
    indices = indices_float.astype(np.int32)
    _, voxel_labels = np.unique(indices, axis=0, return_inverse=True)
    df = pd.DataFrame(data=np.concatenate([voxel_labels[:, np.newaxis], coords], axis=1), columns=np.arange(C+1))
    voxel_mean_df = df.groupby(0).mean()
    samples = voxel_mean_df.to_numpy()

    return samples

def pass_through_filter(coords:np.ndarray, filter_range:np.ndarray) -> np.ndarray:
    """PCL PassThrough filter

    Args:
        coords: coords (N, C)
        filter_range: range (C, 2), `2` is (min, max) 
    
    Returns:
        filted coords: (M, C)
    """

    assert coords.shape[1] == filter_range.shape[0]
    for i, f in enumerate(filter_range):
        fr_min, fr_max = f
        coords = coords[coords[:, i] >= fr_min]
        coords = coords[coords[:, i] <= fr_max]

    return coords