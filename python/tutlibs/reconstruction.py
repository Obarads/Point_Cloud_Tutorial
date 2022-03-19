from typing import Tuple
import numpy as np

from .operator import gather, projection
from .distance import hausdorff_distance
from .nns import k_nearest_neighbors
from .normal_estimation import normal_estimation_v2, normal_orientation
from .tables import TriTable, VertexTable


def ball_pivoting(point_cloud: np.ndarray):

    return


def marching_cubes(voxels: np.ndarray):
    voxel_indices = voxel_to_point(voxels).astype(np.int32)
    Nx, Ny, Nz = voxels.shape
    voxel_vertices = np.zeros((Nx + 1, Ny + 1, Nz + 1))
    voxel_vertices[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2]
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0] + 1, voxel_indices[:, 1], voxel_indices[:, 2]
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0] + 1, voxel_indices[:, 1] + 1, voxel_indices[:, 2]
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0], voxel_indices[:, 1] + 1, voxel_indices[:, 2]
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0], voxel_indices[:, 1], voxel_indices[:, 2] + 1
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0] + 1, voxel_indices[:, 1], voxel_indices[:, 2] + 1
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0] + 1,
        voxel_indices[:, 1] + 1,
        voxel_indices[:, 2] + 1,
    ] = 1
    voxel_vertices[
        voxel_indices[:, 0], voxel_indices[:, 1] + 1, voxel_indices[:, 2] + 1
    ] = 1

    x_arange = np.tile(np.arange(Nx), (Ny * Nz))
    y_arange = np.tile(
        np.tile(np.arange(Ny)[:, np.newaxis], (1, Nx)).reshape(-1), (Nz)
    )
    z_arange = np.tile(np.arange(Nz)[:, np.newaxis], (1, Nx * Ny)).reshape(-1)
    voxel_arange = np.stack([x_arange, y_arange, z_arange], axis=-1)

    tri_table_indices = (
        voxel_vertices[x_arange, y_arange, z_arange] * (2 ** 0)
        + voxel_vertices[x_arange + 1, y_arange, z_arange] * (2 ** 1)
        + voxel_vertices[x_arange + 1, y_arange + 1, z_arange] * (2 ** 2)
        + voxel_vertices[x_arange, y_arange + 1, z_arange] * (2 ** 3)
        + voxel_vertices[x_arange, y_arange, z_arange + 1] * (2 ** 4)
        + voxel_vertices[x_arange + 1, y_arange, z_arange + 1] * (2 ** 5)
        + voxel_vertices[x_arange + 1, y_arange + 1, z_arange + 1] * (2 ** 6)
        + voxel_vertices[x_arange, y_arange + 1, z_arange + 1] * (2 ** 7)
    ).astype(np.int32)

    tri_indices = np.tile(
        np.arange(Nx * Ny * Nz)[:, np.newaxis], (1, len(TriTable[0]))
    )
    tri_indices = tri_indices.reshape(-1)
    rows = TriTable[tri_table_indices]
    rows = rows.reshape(-1)
    mask = rows != -1

    rows = rows[mask]
    tri_indices = tri_indices[mask]
    voxel_arange = voxel_arange[tri_indices]

    vertices = VertexTable[rows]
    vertices += voxel_arange
    triangles = np.arange(len(vertices)).reshape(-1, 3)

    return vertices, triangles


def mesh_to_point(
    vertices: np.ndarray,
    triangles: np.ndarray,
    num_samples: int,
    return_triangle_indices: bool = False,
):
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

    # compute points on faces
    uvw = np.random.rand(num_samples, 3)
    uvw /= np.sum(uvw, axis=1, keepdims=True)
    point_cloud = np.sum(uvw[:, :, np.newaxis] * triangle_vertices, axis=1)

    if return_triangle_indices:
        return point_cloud, triangle_indices
    else:
        return point_cloud


def mesh_to_voxel(
    vertices: np.ndarray, triangles: np.ndarray, voxel_size: float
):
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
    x_idxs = np.tile(
        side_idxs[np.newaxis, np.newaxis, :], (side_len, side_len, 1)
    )
    y_idxs = np.transpose(x_idxs, (1, 2, 0))
    z_idxs = np.transpose(x_idxs, (2, 0, 1))
    sides_idxs = np.stack((x_idxs, y_idxs, z_idxs), axis=-1).astype(np.float32)

    point_cloud = sides_idxs[voxel.astype(np.bool8)]

    return point_cloud


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
    voxels = np.zeros(
        (voxel_length, voxel_length, voxel_length), dtype=np.uint8
    )
    voxels[
        voxel_grid_indices[:, 0],
        voxel_grid_indices[:, 1],
        voxel_grid_indices[:, 2],
    ] = 1
    return voxels


def point_to_mesh(coords: np.ndarray, points: np.ndarray):
    """Create a mesh model from a point cloud.

    Args:
        coords: a point cloud coordinates, (N, 3)

    Returns:
        vertices: (V, 3)
        triangles: (T, 3)

    Reference:
        Hoppe, Hugues, Tony DeRose, Tom Duchamp, John McDonald, and Werner Stuetzle. 1992. “Surface Reconstruction from Unorganized Points.” In Proceedings of the 19th Annual Conference on Computer Graphics and Interactive Techniques - SIGGRAPH ’92. New York, New York, USA: ACM Press. https://doi.org/10.1145/133994.134011.
    """
    k = 10
    rho = 1
    delta = 1
    knn_indices, _ = k_nearest_neighbors(coords, coords, k)
    knn_coords = gather(coords, knn_indices)
    centers = np.mean(knn_coords, axis=1)
    normals = normal_estimation_v2(centers, coords, k=k)
    normals = normal_orientation(centers, normals)

    knn_center_indices, _ = k_nearest_neighbors(points, centers, 1)
    knn_center_indices = knn_center_indices[:, 0]
    closest_centers = centers[knn_center_indices]
    closest_normals = normals[knn_center_indices]

    z_list = projection(points, closest_centers, closest_normals)
    results = []
    for i in range(len(z_list)):
        z = np.array([z_list[i]])
        d = hausdorff_distance(z, coords)
        if d < rho + delta:
            fp = np.matmul(
                (points[i] - closest_centers[i])[:, np.newaxis],
                closest_normals[i][np.newaxis, :],
            )
            results.append(fp)
        else:
            results.append(-1)

    return np.array(fp)


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


