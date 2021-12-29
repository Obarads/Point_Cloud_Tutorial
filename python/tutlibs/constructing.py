from typing import Tuple
import numpy as np
from .transformation import TransformationMatrix as tm
from tutlibs.operator import gather


def marching_cubes():
    return


def tsdf():
    return


def mesh_to_point(
    vertices: np.ndarray, triangles: np.ndarray, num_samples: int
) -> np.ndarray:
    """Sample a point cloud from a mesh data.

    Args:
        vertices: (V, 3)
        triangles: (T, 3)
        num_samples: number of samples

    Return:
        point cloud, (num_samples, 3)
    """

    triangle_vertices = gather(vertices, triangles)  # shape: (T, 3, 3)

    # select triangle meshs
    triangle_areas = (
        np.linalg.norm(
            np.cross(
                triangle_vertices[:, 0] - triangle_vertices[:, 1],
                triangle_vertices[:, 2] - triangle_vertices[:, 1],
            ),
            ord=2,
            axis=1,
        )
        / 2
    )
    triangle_choice_weights = triangle_areas / np.sum(triangle_areas)
    num_triangles = len(triangles)
    triangle_indices = np.random.choice(
        num_triangles, num_samples, p=triangle_choice_weights
    )
    triangle_vertices = triangle_vertices[triangle_indices]

    # compute points on triangle meshs
    uvw = np.random.rand(num_samples, 3)
    uvw /= np.sum(uvw, axis=1, keepdims=True)
    point_cloud = np.sum(uvw[:, :, np.newaxis] * triangle_vertices, axis=1)

    return point_cloud


def mesh_to_voxel(vertices: np.ndarray, triangles: np.ndarray, voxel_size: float):
    # for triangle in triangles:
    #     v3 = vertices[triangle]
    #     v3_min = np.min(v3, axis=0)
    #     v3_max = np.max(v3, axis=0)
    return


def voxel_to_point(voxel: np.ndarray) -> np.ndarray:
    """Construct a point cloud from voxels.

    Args:
        voxel: (D, D, D)

    Returns:
        a point_cloud, (N, 3)
    """
    side_len = len(voxel)
    side_idxs = np.arange(side_len)
    x_idxs = np.tile(side_idxs[np.newaxis, np.newaxis, :], (side_len, side_len, 1))
    y_idxs = np.transpose(x_idxs, (1, 2, 0))
    z_idxs = np.transpose(x_idxs, (2, 0, 1))
    sides_idxs = np.stack((x_idxs, y_idxs, z_idxs), axis=-1).astype(np.float32)

    point_cloud = sides_idxs[voxel.astype(np.bool8)]

    max_xyz = np.max(point_cloud, axis=0)
    min_xyz = np.min(point_cloud, axis=0)

    point_cloud -= (max_xyz - min_xyz) / 2 + min_xyz

    return point_cloud


def point_to_image(
    point_cloud: np.ndarray,
    img: np.ndarray,
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
        img: a RGB image, (H, W, 3)
        fx: a x focal length
        fy: a y focal length
        cx: a x principal point
        cy: a y principal point
        rot_mat: a rotation matrix for camera coordinate system
        trans_mat: a translation matrix for camera coordinate system

    Returns:
        a image, (H, W, 3)
        a pixel indices corresponding to the point cloud, (N, 2)
    """

    img_y, img_x, _ = img.shape
    intrinsic_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    rt_mat = np.concatenate([rot_mat, trans_mat[:, np.newaxis]], axis=1)
    pixel_coords = np.matmul(
        np.matmul(intrinsic_matrix, rt_mat),
        np.concatenate([point_cloud, np.full((len(point_cloud), 1), 1)], axis=1).T,
    ).T
    pixel_coords = pixel_coords[:, :2] / pixel_coords[:, 2][:, np.newaxis]
    pixel_coords = np.array([[img_x, img_y]]) - pixel_coords[:, :2]
    pixel_indices = np.ceil(pixel_coords).astype(np.int32)

    pixel_indices = pixel_indices[pixel_indices[:, 0] < img_x]
    pixel_indices = pixel_indices[pixel_indices[:, 0] >= 0]
    pixel_indices = pixel_indices[pixel_indices[:, 1] < img_y]
    pixel_indices = pixel_indices[pixel_indices[:, 1] >= 0]
    img[pixel_indices[:, 1], pixel_indices[:, 0]] = np.array([0, 0, 0], dtype=np.int32)

    return img, pixel_indices


def point_to_voxel(point_cloud: np.ndarray, voxel_size: float) -> np.ndarray:
    """Construct voxels from a point cloud.

    Args:
        point_cloud: a point cloud (N, 3)
        voxel_size: voxel size

    Return:
        voxels (N, N, N)
    """

    voxel_grid_indices = (point_cloud / voxel_size).astype(np.int32)
    min_index = np.min(voxel_grid_indices, axis=0)
    voxel_grid_indices -= min_index
    voxel_length = np.max(voxel_grid_indices) + 1
    voxels = np.zeros((voxel_length, voxel_length, voxel_length), dtype=np.uint8)
    voxels[
        voxel_grid_indices[:, 0], voxel_grid_indices[:, 1], voxel_grid_indices[:, 2]
    ] = 1
    return voxels


# def point_to_mesh(point_cloud: np.ndarray):
#     """Construct voxels from a point cloud.

#     Args:
#         vertices: (V, 3)
#         triangles: (T, 3)
#         num_samples: number of samples
#     Return:
#         (N, N, N)
#     """
#     return


def depth_to_point(
    depth_image: np.ndarray, fx: float, fy: float, cx: float, cy: float, S: float = 1
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
        a point cloud, (N, 3)
        a pixel indices corresponding to the point cloud, (N, 2)
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
