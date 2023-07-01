import numpy as np

from ...transformation import Transformation as tr

def rotate_point_cloud(batch_data, axis: str = "y"):
    """Randomly rotate the point clouds to augument the dataset
    rotation is per shape based along up direction

    Args:
        batch_data: batch coordinate data, (B, N, 3)
        axis: rotation axis

    Return:
        rotated_data: rotated data, (B, N, 3)
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for i in range(len(batch_data)):
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotated_data[i] = tr.rotation(batch_data[i], axis, rotation_angle)
    return rotated_data


def jitter_point_cloud(batch_data: np.ndarray, sigma=0.01, clip=0.05):
    """Randomly jitter points. jittering is per point.

    Args:
        batch_data: batch coordinate data, (B, N, 3)

    Return:
        jittered_data: jittered_data, (B, N, 3)
    """
    B, N, C = batch_data.shape
    assert clip > 0
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1 * clip, clip)
    jittered_data += batch_data
    return jittered_data

