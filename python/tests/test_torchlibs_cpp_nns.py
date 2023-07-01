import numpy as np
from os.path import join as opj

import torch
from tutlibs.torchlibs.cpp.nns import (
    k_nearest_neighbor as cpp_k_nearest_neighbor,
)
from tutlibs.torchlibs.nns import k_nearest_neighbors as py_k_nearest_neighbor
from tutlibs.torchlibs.utils import t2n
from tutlibs.io import Points


# def test_k_nearest_neighbor(data_dir_path: str):
#     coords, _, _ = Points.read(opj(data_dir_path, "bunny_pc.ply"))

#     batch_coords = torch.tensor(coords[np.newaxis, :], device=0).transpose(1, 2)
#     batch_center_point = torch.tensor(
#         coords[0][None, None, :], device=0
#     ).transpose(1, 2)
#     k = 10

#     neighbor_indices, dists = cpp_k_nearest_neighbor(
#         batch_center_point, batch_coords, k
#     )
#     neighbor_indices = t2n(neighbor_indices[0, 0])

#     print(neighbor_indices)

#     assert np.all(
#         neighbor_indices
#         == np.array([0, 1816, 3388, 933, 1619, 3894, 927, 270, 2795, 1951])
#     )


# def test_k_nearest_neighbor_grad(data_dir_path: str):
#     coords, _, _ = Points.read(opj(data_dir_path, "bunny_pc.ply"))

#     batch_coords = torch.tensor(
#         coords[np.newaxis, :], device=0, requires_grad=True
#     ).transpose(1, 2)
#     batch_center_point = torch.tensor(
#         coords[0][None, None, :], device=0, requires_grad=True
#     ).transpose(1, 2)
#     k = 10
#     neighbor_indices, dists = cpp_k_nearest_neighbor(
#         batch_center_point, batch_coords, k
#     )
#     g_cpp = torch.autograd.grad(torch.mean(dists), batch_center_point)[0]
#     print("g_cpp:", g_cpp)

#     batch_coords = torch.tensor(
#         coords[np.newaxis, :], device=0, requires_grad=True
#     ).transpose(1, 2)
#     batch_center_point = torch.tensor(
#         coords[0][None, None, :], device=0, requires_grad=True
#     ).transpose(1, 2)
#     k = 10
#     neighbor_indices, dists = py_k_nearest_neighbor(
#         batch_center_point, batch_coords, k
#     )
#     g_py = torch.autograd.grad(torch.mean(dists), batch_center_point)[0]

#     print("g_py:", g_py)


# def py_2_k_nearest_neighbor(batch_center_point, batch_coords, k):
#     batch_center_point = batch_center_point.transpose(1, 2)
#     batch_coords = batch_coords.transpose(1, 2)
#     device = batch_center_point.device

#     B, M, C = batch_center_point.shape
#     _, N, _ = batch_coords.shape

#     neighbor_indices = torch.full(
#         (B, M, k), -1, dtype=torch.int32, device=device
#     )
#     dists = torch.full((B, M, k), torch.inf, dtype=torch.float32, device=device)

#     for b in range(B):
#         for m in range(M):
#             for n in range(N):
#                 point_dist = 0.0
#                 for c in range(C):
#                     c_dist = batch_center_point[b, m, c] - batch_coords[b, n, c]
#                     point_dist += c_dist * c_dist

#                 for i in range(k):
#                     if point_dist < dists[b, m, i]:
#                         for j in range(k - 1, i, -1):
#                             dists[b, m, j] = dists[b, m, j - 1]
#                             neighbor_indices[b, m, j] = neighbor_indices[
#                                 b, m, j - 1
#                             ]
#                         dists[b, m, i] = point_dist
#                         neighbor_indices[b, m, i] = n
#                         break

#     return neighbor_indices, dists


def test_k_nearest_neighbor_grad_2(data_dir_path: str):
    # coords, _, _ = Points.read(opj(data_dir_path, "bunny_pc.ply"))
    print("test_k_nearest_neighbor_grad_2")
    coords = np.array([[3, 2, 1], [5, 11, 4], [5, 11, 7]], dtype=np.float32)
    center_coords = np.array([[3, 2, 1], [7, 12, 15], [5, 11, 4]], dtype=np.float32)
    # center_coords = np.tile(center_coords, (512, 1))
    k = 1

    print("cpp impl.")
    batch_coords = torch.tensor(
        coords[np.newaxis, :], device=0, requires_grad=True
    ).transpose(1, 2)
    batch_center_point = torch.tensor(
        center_coords[None, :], device=0, requires_grad=True
    ).transpose(1, 2)
    neighbor_indices, dists = cpp_k_nearest_neighbor(
        batch_center_point, batch_coords, k
    )
    g_cpp = torch.autograd.grad(torch.mean(dists), batch_center_point)[0]
    print("neighbor_indices:", neighbor_indices)
    print("center_points_g:", g_cpp)
    batch_coords = torch.tensor(
        coords[np.newaxis, :], device=0, requires_grad=True
    ).transpose(1, 2)
    batch_center_point = torch.tensor(
        center_coords[None, :], device=0, requires_grad=True
    ).transpose(1, 2)
    neighbor_indices, dists = cpp_k_nearest_neighbor(
        batch_center_point, batch_coords, k
    )
    g_cpp = torch.autograd.grad(torch.mean(dists), batch_coords)[0]
    print("points_g:", g_cpp)

    # print("py impl.")
    # batch_coords = torch.tensor(
    #     coords[np.newaxis, :], device=0, requires_grad=True
    # ).transpose(1, 2)
    # batch_center_point = torch.tensor(
    #     center_coords[None, :], device=0, requires_grad=True
    # ).transpose(1, 2)
    # neighbor_indices, dists = py_2_k_nearest_neighbor(
    #     batch_center_point, batch_coords, k
    # )
    # g_cpp = torch.autograd.grad(torch.mean(dists), batch_center_point)[0]
    # print("neighbor_indices:", neighbor_indices)
    # print("center_points_g:", g_cpp)
    # batch_coords = torch.tensor(
    #     coords[np.newaxis, :], device=0, requires_grad=True
    # ).transpose(1, 2)
    # batch_center_point = torch.tensor(
    #     center_coords[None, :], device=0, requires_grad=True
    # ).transpose(1, 2)
    # neighbor_indices, dists = py_2_k_nearest_neighbor(
    #     batch_center_point, batch_coords, k
    # )
    # g_cpp = torch.autograd.grad(torch.mean(dists), batch_coords)[0]
    # print("points_g:", g_cpp)
