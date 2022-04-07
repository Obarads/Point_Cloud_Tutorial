"""Sampling"""

import numpy as np
import pandas as pd


def random_sampling(coords: np.ndarray, num_sample: int) -> np.ndarray:
    """Random sampling

    Args:
        coords: xyz coordinates (N, C)
        num_sample: number of sample

    Returns:
        indices: sample indices, (num_sample)
    """

    N, _ = coords.shape

    indices = np.random.choice(N, num_sample, replace=False)
    return indices


def furthest_point_sampling(coords: np.ndarray, num_sample: int) -> np.ndarray:
    """Furthest point sampling

    Args:
        coords: xyz coordinates, (N, 3)
        num_sample: number of sammple

    Returns:
        indices: sample indices, (num_sample)
    """
    N, _ = coords.shape

    min_square_dists = np.full(N, 2 ** 16 - 1, dtype=np.float32)
    sample_indices = np.zeros(num_sample, dtype=np.int32)

    # Get first index
    sample_indices[0] = 0
    for i in range(1, num_sample):
        # compute square distances between coords and previous sample.
        previous_sample = coords[sample_indices[i - 1]]
        relative_coords = (
            coords - previous_sample[np.newaxis, :]
        )  # (N, 3) - (1, 3) -> (N, 3)
        square_dists = np.sum(relative_coords ** 2, axis=1)  # (N, 3) -> (N)

        # update minimum distance between coords and samples.
        min_dist_mask = square_dists < min_square_dists
        min_square_dists[min_dist_mask] = square_dists[min_dist_mask]

        # get new furthest point (sample) index.
        sample_indices[i] = np.argmax(min_square_dists)

    return sample_indices


def voxel_grid_sampling(coords: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid sampling

    Args:
        coords: xyz coordinates (N, 3)
        voxel_size: voxel grid size

    Returns:
        samples: sample coords (M, 3)
    """
    N, C = coords.shape

    # get voxel indices.
    indices_float = coords / voxel_size
    indices = indices_float.astype(np.int32)

    # calculate the average coordinate of the point for each voxel.
    _, voxel_labels = np.unique(indices, axis=0, return_inverse=True)
    df = pd.DataFrame(
        data=np.concatenate([voxel_labels[:, np.newaxis], coords], axis=1),
        columns=np.arange(C + 1),
    )
    voxel_mean_df = df.groupby(0).mean()

    # use average coordinates as samples.
    samples = voxel_mean_df.to_numpy()

    return samples


def range_filter(coords: np.ndarray, filter_range: np.ndarray) -> np.ndarray:
    """Range filter

    Args:
        coords: coords (N, C)
        filter_range: range (C, 2), `2` is (min, max)

    Returns:
        indices of filted coords: (M)
    """

    N, C = coords.shape
    assert C == filter_range.shape[0]

    mask = np.full(N, fill_value=True, dtype=np.bool8)
    for i, f in enumerate(filter_range):
        fr_min, fr_max = f
        mask = mask & (coords[:, i] >= fr_min) & (coords[:, i] <= fr_max)

    indices = np.arange(N, dtype=np.int32)[mask]

    return indices
