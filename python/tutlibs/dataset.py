from genericpath import exists
import os
from os.path import join as opj
import numpy as np
import json
import cv2
import scipy.io
import requests
from requests.models import Response
import zipfile
import glob
from typing import Tuple

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

class Pix3D:
    def __init__(self, dataset_dir_path:str, preprocess:bool=True) -> None:
        self.dataset_dir_path = dataset_dir_path
        self.preprocess = preprocess

        # loading data infomation (ex: category, img_size, voxel_path ... etc.)
        with open(opj(self.dataset_dir_path, 'pix3d.json'), 'r') as f:
            self.info = json.load(f)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx:int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, tuple, dict]:
        info = self.info[idx]
        img =  cv2.cvtColor(
            cv2.imread(opj(self.dataset_dir_path, info["img"])),
            cv2.COLOR_BGR2RGB
        )
        mask = cv2.imread(opj(self.dataset_dir_path, info["mask"]))
        voxel = scipy.io.loadmat(opj(self.dataset_dir_path, info["voxel"]))["voxel"]
        obj = Mesh.read(opj(self.dataset_dir_path, info["model"]))

        if self.preprocess:
            voxel = self.preprocess_voxel(voxel)

        return img, mask, voxel, obj, info

    @staticmethod
    def preprocess_voxel(voxel):
        voxel = np.transpose(voxel, (1, 2, 0))
        return voxel

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

    def __getitem__(self, idx:int):
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
        rgb_image_paths = sorted(glob.glob(opj(rgbd_data_dir_path, "rgb", "*")))
        depth_image_paths = sorted(glob.glob(opj(rgbd_data_dir_path, "depth", "*")))

        # download a mesh ply file.
        # Note: 
        if data_number in self.meshs_data_number_list:
            mesh_data_dir_path = opj(self.dataset_dir_path,
                                     data_number,
                                     "mesh")
            if not os.path.exists(mesh_data_dir_path):
                download_data(
                    url=opj(self.download_domain_url, "mesh", f"{data_number}.ply"),
                    output_dir_path=mesh_data_dir_path,
                )
            obj = Mesh.read(opj(mesh_data_dir_path, f"{data_number}.ply"))
        else:
            obj = None

        return rgb_image_paths, depth_image_paths, obj

        

