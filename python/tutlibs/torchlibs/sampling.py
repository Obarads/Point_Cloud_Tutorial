import torch


def farthest_point_sampling(coords: torch.Tensor, num_samples: int):
    """Furthest point sampling with pytorch

    Args:
        coords: pointcloud data, (B, 3, N)
        num_samples: number of samples

    Return:
        centroids: sampled pointcloud index, (B, num_samples)
    """
    coords = coords.transpose(1, 2)

    device = coords.device
    B, N, _ = coords.shape

    min_square_dists = torch.full(
        (B, N), 2**16 - 1, dtype=torch.float32, device=device
    )
    sample_indices = torch.zeros(
        (B, num_samples), dtype=torch.long, device=device
    )

    # Get first index
    sample_indices[:, 0] = 0
    batch_indices = torch.arange(0, B, dtype=torch.long)
    for i in range(1, num_samples):
        # compute square distances between coords and previous sample.
        previous_sample = coords[
            batch_indices, sample_indices[:, i - 1]
        ]  # (B,.N) -> (B)
        relative_coords = (
            coords - previous_sample[:, None, :]
        )  # (B, N, 3) - (B, 1, 3) -> (B, N, 3)
        square_dists = torch.sum(
            relative_coords**2, axis=2
        )  # (B, N, 3) -> (B, N)

        # update minimum distance between coords and samples.
        min_dist_mask = square_dists < min_square_dists
        min_square_dists[min_dist_mask] = square_dists[min_dist_mask]

        # get new furthest point (sample) index.
        sample_indices[:, i] = torch.argmax(min_square_dists, axis=1)

    return sample_indices
