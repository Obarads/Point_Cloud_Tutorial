from glob import glob
import h5py
import os
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset

from ...dataset import download_data

class ModelNet40DatasetForPointNet(Dataset):
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
            data_paths = [
                os.path.join(
                    dataset_dir_path,
                    "modelnet40_ply_hdf5_2048",
                    os.path.basename(path),
                )
                for path in f.read().split("\n")
                if path[-3:] == ".h5"
            ]

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


class ModelNet10DatasetForVoxNet(Dataset):
    """ModelNet40 dataset used in VoxNet experiment"""

    class_id_to_name = {
        0: "bathtub",
        1: "bed",
        2: "chair",
        3: "desk",
        4: "dresser",
        5: "monitor",
        6: "night_stand",
        7: "sofa",
        8: "table",
        9: "toilet",
    }
    class_name_to_id = {v: k for k, v in class_id_to_name.items()}
    class_names = set(class_id_to_name.values())

    def __init__(self, dataset_dir_path: str, mode: str = "train") -> None:

        if not os.path.exists(dataset_dir_path):
            download_data(
                "http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip",
                dataset_dir_path,
                extract_zip=True,
                remove_zip=True,
                verify=False,
            )

        # define data list from loaded volumetric data and labels
        volumetric_dir_path = os.path.join(
            dataset_dir_path, "3DShapeNets", "volumetric_data"
        )
        data_list = []
        label_list = []
        for class_id in self.class_id_to_name:
            data_paths = glob(
                os.path.join(
                    volumetric_dir_path,
                    self.class_id_to_name[class_id],
                    "30",
                    mode,
                    "*",
                )
            )
            for data_path in data_paths:
                volumetric_data = loadmat(data_path)
                data_list.append(volumetric_data["instance"])
                label_list.append(class_id)

        data_list = np.array(data_list, dtype=np.float32)
        label_list = np.array(label_list, dtype=np.int32)

        # add pads
        temp = np.zeros((len(data_list), 32, 32, 32), dtype=np.float32)
        temp[:, 1:-1, 1:-1, 1:-1] = data_list
        data_list = temp

        # add feature axis
        data_list = data_list[:, :, :, :, np.newaxis]

        self.data = data_list
        self.labels = label_list
        self.dataset_dir_path = dataset_dir_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        return data, label


