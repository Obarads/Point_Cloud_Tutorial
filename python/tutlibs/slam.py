import numpy as np
from typing import List

from .registration import icp
from .transformation import TransformationMatrix as tm


def sequential_registration(
    frames: List[np.ndarray], iteration: int = 50
) -> List[np.ndarray]:
    """Estimate sensor poses from a sequence of point cloud frames.

    This is the front-end (odometry) of a point cloud SLAM. For each pair of
    consecutive frames, the relative motion is estimated with ICP, and the
    relative motions are accumulated to obtain the global pose of every frame.
    The first frame defines the origin of the trajectory.

    Args:
        frames: a list of point cloud frames, [(N_0, 3), (N_1, 3), ...]
        iteration: number of ICP iterations for each frame pair

    Returns:
        poses: estimated sensor pose (frame-to-world transform) of every frame,
            [(4, 4), ...]. ``poses[0]`` is the identity matrix.
    """
    # the first frame is the origin of the trajectory.
    poses: List[np.ndarray] = [np.identity(4)]

    for i in range(1, len(frames)):
        # estimate the relative transform that aligns frame i onto frame i-1.
        relative_trans_mat = icp(frames[i], frames[i - 1], iteration=iteration)

        # accumulate the relative transform to get the global pose of frame i.
        pose = tm.composite([poses[i - 1], relative_trans_mat])
        poses.append(pose)

    return poses


def build_map(
    frames: List[np.ndarray], poses: List[np.ndarray]
) -> np.ndarray:
    """Merge frames into a single global map using their poses.

    Args:
        frames: a list of point cloud frames, [(N_0, 3), (N_1, 3), ...]
        poses: frame-to-world transform of every frame, [(4, 4), ...]

    Returns:
        global_map: merged point cloud in the world coordinate, (sum(N_i), 3)
    """
    world_frames = [
        tm.transformation(frame, pose) for frame, pose in zip(frames, poses)
    ]
    return np.concatenate(world_frames, axis=0)


def trajectory(poses: List[np.ndarray]) -> np.ndarray:
    """Get the sensor position of every pose.

    Args:
        poses: frame-to-world transform of every frame, [(4, 4), ...]

    Returns:
        positions: sensor position of every frame, (len(poses), 3)
    """
    # the translation part of a pose is the sensor position in the world.
    return np.stack([pose[:3, 3] for pose in poses], axis=0)
