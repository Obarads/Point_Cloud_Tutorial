import dataclasses
from genericpath import exists
import os
from os.path import join as opj
import numpy as np
import json
import re
import cv2
import scipy.io
import requests
from requests.models import Response
import zipfile
import glob
from typing import List, Tuple
from dataclasses import dataclass

from .io import Mesh

def download_data(url:str, output_dir_path:str, extract_zip:bool=False, remove_zip:bool=False):
    # download
    file_name = os.path.basename(url)
    output_file_path = opj(output_dir_path, file_name)
    if os.path.exists(output_file_path):
        print(f"skip download of {url}")
    else:
        os.makedirs(output_dir_path, exist_ok=True)
        response: Response = requests.get(url, stream=True)
        with open(output_file_path, 'wb') as fd:
            for chunk in response.iter_content(2048):
                fd.write(chunk)

    # extract
    if extract_zip:
        with zipfile.ZipFile(output_file_path) as existing_zip:
            existing_zip.extractall(output_dir_path)

        # remove
        if remove_zip:
            os.system('rm %s' % ('"'+output_file_path+'"'))


@dataclass
class Pix3DData:
    image: np.ndarray
    mask: np.ndarray
    voxel: np.ndarray
    mesh: list # [vertices, triangle_indices, other_data]
    info: dict # pix3d.json dictionary

class Pix3D:
    def __init__(self, dataset_dir_path:str) -> None:
        self.dataset_dir_path = dataset_dir_path

        # loading data infomation (ex: category, img_size, voxel_path ... etc.)
        with open(opj(self.dataset_dir_path, 'pix3d.json'), 'r') as f:
            self.info = json.load(f)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx:int) -> Pix3DData:
        data = Pix3DData
        data.info = self.info[idx]
        data.image =  cv2.cvtColor(
            cv2.imread(opj(self.dataset_dir_path, data.info["img"])),
            cv2.COLOR_BGR2RGB
        )
        data.mask = cv2.imread(opj(self.dataset_dir_path, data.info["mask"]))
        data.voxel = np.transpose(
            scipy.io.loadmat(opj(self.dataset_dir_path, data.info["voxel"]))["voxel"],
            (1, 2, 0)
        )
        data.mesh = Mesh.read(opj(self.dataset_dir_path, data.info["model"]))

        return data


@dataclass
class Redwood3DScanData:
    color_image_paths: list
    depth_image_paths: list
    mesh: list # [vertices, triangle_indices, other_data]

class Redwood3DScan:
    download_domain_url = "https://s3.us-west-1.wasabisys.com/redwood-3dscan"

    def __init__(self, datalist_dir_path:str, dataset_dir_path:str) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.datalist_dir_path = datalist_dir_path

        with open(opj(self.datalist_dir_path, 'rgbds.json')) as f:
            self.rgbd_data_number_list = json.load(f)
        with open(opj(self.datalist_dir_path, 'meshes.json')) as f:
            self.meshs_data_number_list = json.load(f)
        # with open(opj(self.datalist_dir_path, 'categories.json')) as f:
        #     self.categories_data_number_dict = json.load(f)

    def __len__(self):
        return len(self.rgbd_data_number_list)

    def __getitem__(self, idx:int) -> Redwood3DScanData:
        data = Redwood3DScanData
        data_number = self.rgbd_data_number_list[idx]

        # download RGB-D images.
        rgbd_data_dir_path = opj(self.dataset_dir_path,
                                 data_number,
                                 "rgbd")
        if not os.path.exists(rgbd_data_dir_path):
            download_data(
                url=opj(self.download_domain_url, "rgbd", f"{data_number}.zip"),
                output_dir_path=rgbd_data_dir_path,
                extract_zip=True,
                remove_zip=True
            )
        data.color_image_paths = sorted(glob.glob(opj(rgbd_data_dir_path, "rgb", "*")))
        data.depth_image_paths = sorted(glob.glob(opj(rgbd_data_dir_path, "depth", "*")))

        # download a mesh ply file.
        # Note: some samples have the mesh data
        if data_number in self.meshs_data_number_list:
            mesh_data_dir_path = opj(self.dataset_dir_path,
                                     data_number,
                                     "mesh")
            if not os.path.exists(mesh_data_dir_path):
                download_data(
                    url=opj(self.download_domain_url, "mesh", f"{data_number}.ply"),
                    output_dir_path=mesh_data_dir_path,
                )
            data.mesh = Mesh.read(opj(mesh_data_dir_path, f"{data_number}.ply"))
        else:
            data.mesh = None

        return data


@dataclass
class ScanNetData:
    """__getitem__ return values of ScanNet class"""
    scan_data_path: str

    # mesh data
    vh_clean_2_mesh: list # [vertices, triangle_indices]
    vh_clean_2_label_mesh: list # [vertices, triangle_indices, other_data]

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
    def __init__(self, dataset_dir_path:str) -> None:
        """ScanNet Dataset

        Args:
            dataset_dir_path: a path to folder containing scene%04d_%02d (ex: scene0000_02) folders.
        """
        self.dataset_dir_path = dataset_dir_path
        self.scan_data_path_list = sorted(glob.glob(opj(self.dataset_dir_path, "*")))

    def __len__(self):
        return len(self.scan_data_path_list)

    def __getitem__(self, idx:int) -> ScanNetData:
        # get path to scan data folder (/path/to/scene%04d_%02d)
        scan_data_path = self.scan_data_path_list[idx]
        data = ScanNetData

        data.scan_data_path = scan_data_path

        data.vh_clean_2_mesh = Mesh.read(opj(scan_data_path, 'scene0000_00_vh_clean_2.ply'))
        data.vh_clean_2_label_mesh = Mesh.read(opj(scan_data_path, 'scene0000_00_vh_clean_2.labels.ply'))

        def file_number(path:str):
            file_number = path.split("/")[-1].split('.')[0]
            return int(file_number)

        data.depth_image_paths = sorted(glob.glob(opj(scan_data_path, "sens", "depth", "*")), key=file_number)
        data.depth_image_intrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "intrinsic_depth.txt"))
        data.depth_image_extrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "extrinsic_depth.txt"))
        data.color_image_paths = sorted(glob.glob(opj(scan_data_path, "sens", "color", "*")), key=file_number)
        data.color_image_intrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "intrinsic_color.txt"))
        data.color_image_extrinsic_matrix = np.loadtxt(opj(scan_data_path, "sens", "intrinsic", "extrinsic_color.txt"))

        data.pose_file_paths = sorted(glob.glob(opj(scan_data_path, "sens", "pose", "*")), key=file_number)

        assert len(data.depth_image_paths) == len(data.color_image_paths) and len(data.depth_image_paths) == len(data.pose_file_paths)

        return data


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