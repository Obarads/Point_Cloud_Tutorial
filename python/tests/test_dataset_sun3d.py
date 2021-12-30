import numpy as np
from os.path import join as opj

from tutlibs.dataset import SUN3D


def test_init(data_dir_path: str):
    dataset_dir_path = opj(data_dir_path, "sun3d")
    datalist_path = opj(dataset_dir_path, "SUN3Dv1.txt")
    dataset = SUN3D(dataset_dir_path, datalist_path)
    data = dataset[0]
    print(data.intrinsics_matrix)
