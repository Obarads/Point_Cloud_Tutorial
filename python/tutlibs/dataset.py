import os
from os.path import join as opj
import numpy as np
import json
import cv2
import scipy.io
import requests
import re
from requests.models import Response
import zipfile
import glob
from typing import List, Tuple
from dataclasses import dataclass

from .io import Mesh


def download_data(
    url: str, output_dir_path: str, extract_zip: bool = False, remove_zip: bool = False
):
    # download
    file_name = os.path.basename(url)
    output_file_path = opj(output_dir_path, file_name)
    if os.path.exists(output_file_path):
        print(f"skip download of {url}")
    else:
        os.makedirs(output_dir_path, exist_ok=True)
        response: Response = requests.get(url, stream=True)
        with open(output_file_path, "wb") as fd:
            for chunk in response.iter_content(2048):
                fd.write(chunk)

    # extract
    if extract_zip:
        with zipfile.ZipFile(output_file_path) as existing_zip:
            existing_zip.extractall(output_dir_path)

        # remove
        if remove_zip:
            os.system("rm %s" % ('"' + output_file_path + '"'))


@dataclass
class Pix3DData:
    image: np.ndarray
    mask: np.ndarray
    voxel: np.ndarray
    mesh: list  # [vertices, triangle_indices, other_data]
    info: dict  # pix3d.json dictionary


class Pix3D:
    def __init__(self, dataset_dir_path: str) -> None:
        self.dataset_dir_path = dataset_dir_path

        # loading data infomation (ex: category, img_size, voxel_path ... etc.)
        with open(opj(self.dataset_dir_path, "pix3d.json"), "r") as f:
            self.info = json.load(f)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx: int) -> Pix3DData:
        data = Pix3DData
        data.info = self.info[idx]
        data.image = cv2.cvtColor(
            cv2.imread(opj(self.dataset_dir_path, data.info["img"])), cv2.COLOR_BGR2RGB
        )
        data.mask = cv2.imread(opj(self.dataset_dir_path, data.info["mask"]))
        data.voxel = scipy.io.loadmat(opj(self.dataset_dir_path, data.info["voxel"]))[
            "voxel"
        ]
        data.mesh = Mesh.read(opj(self.dataset_dir_path, data.info["model"]))

        return data


@dataclass
class Redwood3DScanData:
    color_image_paths: list
    depth_image_paths: list
    mesh: list  # [vertices, triangle_indices, other_data]


class Redwood3DScan:
    download_domain_url = "https://s3.us-west-1.wasabisys.com/redwood-3dscan"

    def __init__(self, datalist_dir_path: str, dataset_dir_path: str) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.datalist_dir_path = datalist_dir_path

        with open(opj(self.datalist_dir_path, "rgbds.json")) as f:
            self.rgbd_data_number_list = json.load(f)
        with open(opj(self.datalist_dir_path, "meshes.json")) as f:
            self.meshs_data_number_list = json.load(f)
        # with open(opj(self.datalist_dir_path, 'categories.json')) as f:
        #     self.categories_data_number_dict = json.load(f)

    def __len__(self):
        return len(self.rgbd_data_number_list)

    def __getitem__(self, idx: int) -> Redwood3DScanData:
        data = Redwood3DScanData
        data_number = self.rgbd_data_number_list[idx]

        # If dataset_dir_path/data_number/* data is not downloaded, this funciton download it.
        self.download(data_number)

        # get RGB-D images.
        rgbd_data_dir_path = opj(self.dataset_dir_path, data_number, "rgbd")
        data.color_image_paths = sorted(glob.glob(opj(rgbd_data_dir_path, "rgb", "*")))
        data.depth_image_paths = sorted(
            glob.glob(opj(rgbd_data_dir_path, "depth", "*"))
        )

        # get a Mesh data.
        # Note: some samples have the mesh data (therefore, it check self.meshs_data_number_list)
        if data_number in self.meshs_data_number_list:
            mesh_data_dir_path = opj(self.dataset_dir_path, data_number, "mesh")
            data.mesh = Mesh.read(opj(mesh_data_dir_path, f"{data_number}.ply"))
        else:
            data.mesh = None

        return data

    def download(
        self,
        data_number: str,
    ) -> None:
        # download RGB-D datasets.
        rgbd_data_dir_path = opj(self.dataset_dir_path, data_number, "rgbd")
        if not os.path.exists(rgbd_data_dir_path):
            download_data(
                url=opj(self.download_domain_url, "rgbd", f"{data_number}.zip"),
                output_dir_path=rgbd_data_dir_path,
                extract_zip=True,
                remove_zip=True,
            )

        # download a mesh ply file.
        # Note: some samples have the mesh data (therefore, it check self.meshs_data_number_list)
        mesh_data_dir_path = opj(self.dataset_dir_path, data_number, "mesh")
        data_number in self.meshs_data_number_list
        if (not os.path.exists(rgbd_data_dir_path)) and (
            data_number in self.meshs_data_number_list
        ):
            if not os.path.exists(mesh_data_dir_path):
                download_data(
                    url=opj(self.download_domain_url, "mesh", f"{data_number}.ply"),
                    output_dir_path=mesh_data_dir_path,
                )


@dataclass
class ScanNetData:
    """__getitem__ return values of ScanNet class"""

    scan_data_path: str

    # mesh data
    vh_clean_2_mesh: list  # [vertices, triangle_indices]
    vh_clean_2_label_mesh: list  # [vertices, triangle_indices, other_data]

    # image paths, intrinsic matrixes and extrinsic matrixes
    depth_image_paths: List[str]
    depth_image_intrinsic_matrix: np.ndarray
    depth_image_extrinsic_matrix: np.ndarray
    color_image_paths: List[str]
    color_image_intrinsic_matrix: np.ndarray
    color_image_extrinsic_matrix: np.ndarray

    # paths to pose (transformation) matrix file
    pose_file_paths: List[str]


class ScanNet:
    def __init__(self, dataset_dir_path: str) -> None:
        """ScanNet Dataset

        Args:
            dataset_dir_path: a path to folder containing scene%04d_%02d (ex: scene0000_02) folders.
        """
        self.dataset_dir_path = dataset_dir_path
        self.scan_data_path_list = sorted(glob.glob(opj(self.dataset_dir_path, "*")))

    def __len__(self):
        return len(self.scan_data_path_list)

    def __getitem__(self, idx: int) -> ScanNetData:
        # get path to scan data folder (/path/to/scene%04d_%02d)
        scan_data_path = self.scan_data_path_list[idx]
        data = ScanNetData

        data.scan_data_path = scan_data_path

        data.vh_clean_2_mesh = Mesh.read(
            opj(scan_data_path, "scene0000_00_vh_clean_2.ply")
        )
        data.vh_clean_2_label_mesh = Mesh.read(
            opj(scan_data_path, "scene0000_00_vh_clean_2.labels.ply")
        )

        def file_number(path: str):
            file_number = path.split("/")[-1].split(".")[0]
            return int(file_number)

        data.depth_image_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "depth", "*")), key=file_number
        )
        data.depth_image_intrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "intrinsic_depth.txt")
        )
        data.depth_image_extrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "extrinsic_depth.txt")
        )
        data.color_image_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "color", "*")), key=file_number
        )
        data.color_image_intrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "intrinsic_color.txt")
        )
        data.color_image_extrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "extrinsic_color.txt")
        )

        data.pose_file_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "pose", "*")), key=file_number
        )

        assert len(data.depth_image_paths) == len(data.color_image_paths) and len(
            data.depth_image_paths
        ) == len(data.pose_file_paths)

        return data


@dataclass
class SUN3DData:
    """__getitem__ return values of SUN3D class"""

    data_path: str
    color_image_paths: list
    depth_image_paths: list
    annotation_file_paths: list
    extrinsics_file_paths: list
    intrinsics_matrix: list


class SUN3D:
    path_list_url = "http://sun3d.cs.princeton.edu/SUN3Dv1.txt"
    download_domain_url = "http://sun3d.cs.princeton.edu/data/"

    def __init__(self, dataset_dir_path, datalist_path) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.datalist_path = datalist_path

        with open(self.datalist_path) as f:
            self.datalist = f.read().split("\n")

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx: int) -> SUN3DData:
        data = SUN3DData
        data_path = self.datalist[idx]
        data.data_path = data_path

        # If dataset_dir_path/data_path/* data is not downloaded, this funciton download it.
        self.download(data_path)

        # get RGB-D images.
        data.color_image_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "image", "*"))
        )
        data.depth_image_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "depth", "*"))
        )

        # get other data.
        data.annotation_file_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "annotation", "*"))
        )
        data.extrinsics_file_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "extrinsics", "*"))
        )
        data.intrinsics_matrix = np.loadtxt(
            opj(self.dataset_dir_path, data_path, "intrinsics.txt")
        )

        return data

    def download(self, data_folder_path):
        folder_url_dict = {
            "annotation": ".json",
            "depth": ".png",
            "extrinsics": ".txt",
            "image": ".jpg",
        }
        for key in folder_url_dict:
            ext = folder_url_dict[key]
            data_file_table_url = opj(self.download_domain_url, data_folder_path, key)
            response: Response = requests.get(data_file_table_url)
            response.encoding = response.apparent_encoding
            filename_list = re.findall(
                f"(?<=href\=[\"'])[^\"']*{ext}[^\"']*(?=[\"'])", response.text
            )
            for filename in filename_list:
                output_dir_path = opj(self.dataset_dir_path, data_folder_path, key)
                output_file_path = opj(output_dir_path, filename)
                if not os.path.exists(output_file_path):
                    download_file_url = opj(data_file_table_url, filename)
                    download_data(download_file_url, output_dir_path)

        intrinsics_file_path = opj(
            self.dataset_dir_path, data_folder_path, "intrinsics.txt"
        )
        if not os.path.exists(intrinsics_file_path):
            intrinsics_url = opj(
                self.download_domain_url, data_folder_path, "intrinsics.txt"
            )
            download_data(intrinsics_url, opj(self.dataset_dir_path, data_folder_path))


# @dataclass
# class KITTIData:
#     """__getitem__ return values of ScanNet class"""
#     scan_data_path: str

# class KITTI:
#     def __init__(self, dataset_path:str) -> None:
#         self.dataset_path = dataset_path

#     def __len__(self):
#         return

#     def __getitem__(self, idx:int) -> ScanNetData:
#         # get path to scan data folder (/path/to/scene%04d_%02d)
#         scan_data_path = self.scan_data_path_list[idx]
#         data = ScanNetData

#         data.scan_data_path = scan_data_path

#         data.vh_clean_2_mesh = Mesh.read(opj(scan_data_path, 'scene0000_00_vh_clean_2.ply'))
#         data.vh_clean_2_label_mesh = Mesh.read(opj(scan_data_path, 'scene0000_00_vh_clean_2.labels.ply'))

#         def file_number(path:str):
#             file_number = path.split("/")[-1].split('.')[0]
#             return int(file_number)

#         data.depth_image_paths = sorted(glob.glob(opj(scan_data_path, "sens", "depth", "*")), key=file_number)
#         data.depth_image_intrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "intrinsic_depth.txt"))
#         data.depth_image_extrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "extrinsic_depth.txt"))
#         data.color_image_paths = sorted(glob.glob(opj(scan_data_path, "sens", "color", "*")), key=file_number)
#         data.color_image_intrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "intrinsic_color.txt"))
#         data.color_image_extrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "extrinsic_color.txt"))

#         data.pose_file_paths = sorted(glob.glob(opj(scan_data_path, "sens", "pose", "*")), key=file_number)

#         assert len(data.depth_image_paths) == len(data.color_image_paths) and len(data.depth_image_paths) == len(data.pose_file_paths)

#         return data
