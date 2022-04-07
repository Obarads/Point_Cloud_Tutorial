import numpy as np
from typing import Tuple


def point_to_image(
    point_cloud: np.ndarray,
    height: int,
    width: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    rot_mat: np.ndarray,
    trans_mat: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project a point cloud to an image.

    Args:
        point_cloud: a point cloud, (N, 3)
        height: image height
        width: image width
        fx: a x focal length
        fy: a y focal length
        cx: a x principal point
        cy: a y principal point
        rot_mat: a rotation matrix for camera coordinate system
        trans_mat: a translation matrix for camera coordinate system

    Returns:
        img: a image, (H, W, 3)
        pixel_indices: a pixel indices corresponding to the point cloud, (N, 2)
    """

    img = np.full((height, width, 3), 255, dtype=np.int32)
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    rt_mat = np.concatenate([rot_mat, trans_mat[:, np.newaxis]], axis=1)
    pixel_coords = np.matmul(
        np.matmul(intrinsic_matrix, rt_mat),
        np.concatenate([point_cloud, np.full((len(point_cloud), 1), 1)], axis=1).T,
    ).T
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2][:, np.newaxis]
    pixel_coords = np.array([[width, height]]) - pixel_coords[:, :2]
    pixel_indices = np.ceil(pixel_coords).astype(np.int32)

    pixel_indices = pixel_indices[pixel_indices[:, 0] < width]
    pixel_indices = pixel_indices[pixel_indices[:, 0] >= 0]
    pixel_indices = pixel_indices[pixel_indices[:, 1] < height]
    pixel_indices = pixel_indices[pixel_indices[:, 1] >= 0]
    img[pixel_indices[:, 1], pixel_indices[:, 0]] = np.array([0, 0, 0], dtype=np.int32)

    return img, pixel_indices


def depth_to_point(
    depth_image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    S: float = 1,
):
    """Construct a point cloud from a depth image.

    Args:
        depth_image: a depth image, (H, W)
        fx: a x focal length
        fy: a y focal length
        cx: a x principal point
        cy: a y principal point
        S: skews

    Returns:
        a point cloud: (N, 3)
        a pixel indices corresponding to the point cloud: (N, 2)
    """

    img_y, img_x = depth_image.shape
    inverse_intrinsic_matrix = np.array(
        [
            [1 / fx, -S / (fx * fy), (S * cy - cx * fy) / (fx * fy), 0],
            [0, 1 / fy, -cy / fy, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    pixel_coords = (
        np.concatenate(
            [
                np.tile(np.arange(img_x)[:, np.newaxis, np.newaxis], (1, img_y, 1)),
                np.tile(np.arange(img_y)[np.newaxis, :, np.newaxis], (img_x, 1, 1)),
                np.full((img_x, img_y, 2), fill_value=1),
            ],
            axis=2,
        )
        .astype(np.float32)
        .reshape(-1, 4)
    )

    depths = depth_image.T.reshape(-1, 1)
    point_cloud = np.matmul(inverse_intrinsic_matrix, pixel_coords.T).T * depths
    pixel_indices = pixel_coords[:, :2].astype(np.int32)

    return point_cloud[:, :3], pixel_indices
