import numpy as np
from os.path import join as opj

import torch
from tutlibs.torchlibs.cpp.nns import k_nearest_neighbor
from tutlibs.torchlibs.utils import t2n
from tutlibs.io import Points


def test_k_nearest_neighbor(data_dir_path: str):
    coords, _, _ = Points.read(opj(data_dir_path, "bunny_pc.ply"))

    batch_coords = torch.tensor(coords[np.newaxis, :], device=0).transpose(1, 2)
    batch_center_point = torch.tensor(
        coords[0][None, None, :], device=0
    ).transpose(1, 2)
    k = 10

    neighbor_indices = k_nearest_neighbor(batch_center_point, batch_coords, k)
    neighbor_indices = t2n(neighbor_indices[0, 0])

    assert np.all(
        neighbor_indices
        == np.array([0, 1816, 3388, 933, 1619, 3894, 927, 270, 2795, 1951])
    )
