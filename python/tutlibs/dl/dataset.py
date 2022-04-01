import h5py
import os
import numpy as np

from torch.utils.data import Dataset
from ..dataset import download_data
from ..transformation import Transformation as tr

class ModelNet40Dataset(Dataset):
    """ModelNet40 dataset used in PointNet experiment"""

    def __init__(self, dataset_dir_path: str, mode: str = "train") -> None:
        if not os.path.exists(dataset_dir_path):
            download_data(
                "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip",
                dataset_dir_path,
                extract_zip=True,
                remove_zip=True,
                verify=False,
            )

        path_list_file_path = os.path.join(
            dataset_dir_path, "modelnet40_ply_hdf5_2048", f"{mode}_files.txt"
        )
        with open(path_list_file_path, mode="r") as f:
            data_paths = [os.path.join(dataset_dir_path, "modelnet40_ply_hdf5_2048", os.path.basename(path)) for path in f.read().split("\n") if path[-3:] == ".h5"]

        data_list = []
        label_list = []
        for data_path in data_paths:
            with h5py.File(data_path, "r") as f:
                data_list.append(f["data"][:])
                label_list.append(f["label"][:, 0])

        self.data = np.concatenate(data_list, axis=0)
        self.labels = np.concatenate(label_list, axis=0)

        self.dataset_dir_path = dataset_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


def rotate_point_cloud(batch_data, axis:str="y"):
    """ Randomly rotate the point clouds to augument the dataset
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

def jitter_point_cloud(batch_data:np.ndarray, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.

        Args:
            batch_data: batch coordinate data, (B, N, 3)

        Return:
            jittered_data: jittered_data, (B, N, 3)
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data
