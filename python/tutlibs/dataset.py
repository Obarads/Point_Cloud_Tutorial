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

from .io import Mesh, Points


def download_data(
    url: str,
    output_dir_path: str,
    extract_zip: bool = False,
    remove_zip: bool = False,
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
        info = self.info[idx]
        image = cv2.cvtColor(
            cv2.imread(opj(self.dataset_dir_path, info["img"])),
            cv2.COLOR_BGR2RGB,
        )
        mask = cv2.imread(opj(self.dataset_dir_path, info["mask"]))
        voxel = scipy.io.loadmat(
            opj(self.dataset_dir_path, info["voxel"])
        )["voxel"]
        mesh = Mesh.read(opj(self.dataset_dir_path, info["model"]))

        data = Pix3DData(
            info=info, image=image, mask=mask, voxel=voxel, mesh=mesh
        )

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
        data_number = self.rgbd_data_number_list[idx]

        # If dataset_dir_path/data_number/* data is not downloaded, this funciton download it.
        self.download(data_number)

        # get RGB-D images paths.
        rgbd_data_dir_path = opj(self.dataset_dir_path, data_number, "rgbd")
        color_image_paths = sorted(
            glob.glob(opj(rgbd_data_dir_path, "rgb", "*"))
        )
        depth_image_paths = sorted(
            glob.glob(opj(rgbd_data_dir_path, "depth", "*"))
        )

        # get a Mesh data.
        # Note: some samples have the mesh data (therefore, it check self.meshs_data_number_list)
        if data_number in self.meshs_data_number_list:
            mesh_data_dir_path = opj(self.dataset_dir_path, data_number, "mesh")
            mesh = Mesh.read(opj(mesh_data_dir_path, f"{data_number}.ply"))
        else:
            mesh = None

        data = Redwood3DScanData(
            color_image_paths=color_image_paths,
            depth_image_paths=depth_image_paths,
            mesh=mesh,
        )

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
        if (not os.path.exists(mesh_data_dir_path)) and (
            data_number in self.meshs_data_number_list
        ):
            if not os.path.exists(mesh_data_dir_path):
                download_data(
                    url=opj(
                        self.download_domain_url, "mesh", f"{data_number}.ply"
                    ),
                    output_dir_path=mesh_data_dir_path,
                )


@dataclass
class RedwoodIndoorData:
    color_image_paths: list
    clean_depth_image_paths: list
    noisy_depth_image_paths: list
    trajectory_list: list
    point_cloud: np.ndarray  # (N, 3)


class RedwoodIndoor:
    download_domain_url = "http://redwood-data.org/indoor/data/"

    data_name_list = [
        "livingroom1",
        "livingroom2",
        "office1",
        "office2",
    ]

    def __init__(self, dataset_dir_path: str) -> None:
        self.dataset_dir_path = dataset_dir_path

    def __len__(self):
        return len(self.data_name_list)

    def __getitem__(self, idx: int) -> RedwoodIndoorData:
        data_name = self.data_name_list[idx]

        # If dataset_dir_path/data_number/* data is not downloaded, this funciton download it.
        self.download(data_name)

        # get RGB image paths.
        rgb_data_dir_path = opj(self.dataset_dir_path, data_name, "rgb")
        color_image_paths = sorted(glob.glob(opj(rgb_data_dir_path, "*")))

        # get clean depth image paths.
        clean_depth_data_dir_path = opj(
            self.dataset_dir_path, data_name, "clean_depth"
        )
        clean_depth_image_paths = sorted(
            glob.glob(opj(clean_depth_data_dir_path, "*"))
        )

        # get noisy depth image paths.
        noisy_depth_data_dir_path = opj(
            self.dataset_dir_path, data_name, "noisy_depth"
        )
        noisy_depth_image_paths = sorted(
            glob.glob(opj(noisy_depth_data_dir_path, "*"))
        )

        # get trajectory txt.
        trajectory_list = []
        with open(
            opj(self.dataset_dir_path, data_name, f"{data_name}-traj.txt"), "r"
        ) as f:
            contents = [line.split(" ") for line in f.read().split("\n")]
            for i in range(int(len(contents) / 5)):
                index = i * 5
                trajectory_list.append(
                    np.array(
                        [
                            contents[index + 1],
                            contents[index + 2],
                            contents[index + 3],
                            contents[index + 4],
                        ]
                    )
                )
        trajectory_list = np.asarray(trajectory_list, dtype=np.float32)

        # get a point cloud.
        point_cloud = Points.read(
            opj(opj(self.dataset_dir_path, data_name), f"{data_name[:-1]}.ply")
        )

        data = RedwoodIndoorData(
            color_image_paths=color_image_paths,
            clean_depth_image_paths=clean_depth_image_paths,
            noisy_depth_image_paths=noisy_depth_image_paths,
            trajectory_list=trajectory_list,
            point_cloud=point_cloud,
        )

        return data

    def download(
        self,
        data_name: str,
    ) -> None:
        # download a rgb zip.
        rgb_data_dir_path = opj(self.dataset_dir_path, data_name, "rgb")
        if not os.path.exists(rgb_data_dir_path):
            download_data(
                url=opj(self.download_domain_url, f"{data_name}-color.zip"),
                output_dir_path=rgb_data_dir_path,
                extract_zip=True,
                remove_zip=True,
            )

        # download a clean depth zip.
        clean_depth_data_dir_path = opj(
            self.dataset_dir_path, data_name, "clean_depth"
        )
        if not os.path.exists(clean_depth_data_dir_path):
            download_data(
                url=opj(
                    self.download_domain_url, f"{data_name}-depth-clean.zip"
                ),
                output_dir_path=clean_depth_data_dir_path,
                extract_zip=True,
                remove_zip=True,
            )

        # download a noisy depth zip.
        noisy_depth_data_dir_path = opj(
            self.dataset_dir_path, data_name, "noisy_depth"
        )
        if not os.path.exists(noisy_depth_data_dir_path):
            download_data(
                url=opj(
                    self.download_domain_url, f"{data_name}-depth-simulated.zip"
                ),
                output_dir_path=noisy_depth_data_dir_path,
                extract_zip=True,
                remove_zip=True,
            )

        # download a oni zip.
        # oni_data_dir_path = opj(self.dataset_dir_path, data_name, "oni")
        # if not os.path.exists(oni_data_dir_path):
        #     download_data(
        #         url=opj(self.download_domain_url, f"{data_name}-oni.zip"),
        #         output_dir_path=oni_data_dir_path,
        #         extract_zip=True,
        #         remove_zip=True,
        #     )

        # download a Ground-truth Trajectory txt.
        trajectory_data_dir_path = opj(self.dataset_dir_path, data_name)
        if not os.path.exists(
            opj(trajectory_data_dir_path, f"{data_name}-traj.txt")
        ):
            download_data(
                url=opj(self.download_domain_url, f"{data_name}-traj.txt"),
                output_dir_path=trajectory_data_dir_path,
            )

        # download a dense point cloud zip file.
        point_data_dir_path = opj(self.dataset_dir_path, data_name)
        if not os.path.exists(
            opj(point_data_dir_path, f"{data_name[:-1]}.ply")
        ):
            download_data(
                url=opj(self.download_domain_url, f"{data_name[:-1]}.ply.zip"),
                output_dir_path=point_data_dir_path,
                extract_zip=True,
                remove_zip=True,
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
        self.scan_data_path_list = sorted(
            glob.glob(opj(self.dataset_dir_path, "*"))
        )

    def __len__(self):
        return len(self.scan_data_path_list)

    def __getitem__(self, idx: int) -> ScanNetData:
        # get path to scan data folder (/path/to/scene%04d_%02d)
        scan_data_path = self.scan_data_path_list[idx]

        vh_clean_2_mesh = Mesh.read(
            opj(scan_data_path, "scene0000_00_vh_clean_2.ply")
        )
        vh_clean_2_label_mesh = Mesh.read(
            opj(scan_data_path, "scene0000_00_vh_clean_2.labels.ply")
        )

        def file_number(path: str):
            file_number = path.split("/")[-1].split(".")[0]
            return int(file_number)

        depth_image_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "depth", "*")),
            key=file_number,
        )
        depth_image_intrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "intrinsic_depth.txt")
        )
        depth_image_extrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "extrinsic_depth.txt")
        )
        color_image_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "color", "*")),
            key=file_number,
        )
        color_image_intrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "intrinsic_color.txt")
        )
        color_image_extrinsic_matrix = np.loadtxt(
            opj(scan_data_path, "sens", "intrinsic", "extrinsic_color.txt")
        )

        pose_file_paths = sorted(
            glob.glob(opj(scan_data_path, "sens", "pose", "*")), key=file_number
        )

        assert len(data.depth_image_paths) == len(
            data.color_image_paths
        ) and len(data.depth_image_paths) == len(data.pose_file_paths)

        data = ScanNetData(
            scan_data_path=scan_data_path,
            vh_clean_2_mesh=vh_clean_2_mesh,
            vh_clean_2_label_mesh=vh_clean_2_label_mesh,
            depth_image_paths=depth_image_paths,
            depth_image_intrinsic_matrix=depth_image_intrinsic_matrix,
            depth_image_extrinsic_matrix=depth_image_extrinsic_matrix,
            color_image_paths=color_image_paths,
            color_image_intrinsic_matrix=color_image_intrinsic_matrix,
            color_image_extrinsic_matrix=color_image_extrinsic_matrix,
            pose_file_paths=pose_file_paths,
        )

        return data


@dataclass
class SUN3DData:
    """__getitem__ return values of SUN3D class"""

    data_path: str
    color_image_paths: list
    depth_image_paths: list
    annotation_file_paths: list
    extrinsics_file_paths: list
    intrinsics_matrix: np.ndarray


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
        data_path = self.datalist[idx]

        # If dataset_dir_path/data_path/* data is not downloaded, this funciton download it.
        self.download(data_path)

        # get RGB-D images.
        color_image_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "image", "*"))
        )
        depth_image_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "depth", "*"))
        )

        # get other data.
        annotation_file_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "annotation", "*"))
        )
        extrinsics_file_paths = sorted(
            glob.glob(opj(self.dataset_dir_path, data_path, "extrinsics", "*"))
        )
        intrinsics_matrix = np.loadtxt(
            opj(self.dataset_dir_path, data_path, "intrinsics.txt")
        )

        data = SUN3DData(
            data_path=data_path,
            color_image_paths=color_image_paths,
            depth_image_paths=depth_image_paths,
            annotation_file_paths=annotation_file_paths,
            extrinsics_file_paths=extrinsics_file_paths,
            intrinsics_matrix=intrinsics_matrix,
        )

        return data

    def download(self, scene_dir_name):
        scene_data_dir_dict = {
            "annotation": ".json",
            "depth": ".png",
            "extrinsics": ".txt",
            "image": ".jpg",
        }
        remote_scene_dir_url = opj(self.download_domain_url, scene_dir_name)
        local_scene_dir_path = opj(self.dataset_dir_path, scene_dir_name)
        for key in scene_data_dir_dict:
            ext = scene_data_dir_dict[key]
            remote_scene_data_dir_url = opj(remote_scene_dir_url, key)
            loacl_scene_data_dir_path = opj(local_scene_dir_path, key)
            response: Response = requests.get(remote_scene_data_dir_url)
            response.encoding = response.apparent_encoding
            filename_list = re.findall(
                f"(?<=href\=[\"'])[^\"']*{ext}[^\"']*(?=[\"'])", response.text
            )
            for filename in filename_list:
                local_scene_data_file_path = opj(
                    loacl_scene_data_dir_path, filename
                )
                remote_scene_data_file_url = opj(
                    remote_scene_data_dir_url, filename
                )
                if not os.path.exists(local_scene_data_file_path):
                    download_data(
                        remote_scene_data_file_url, loacl_scene_data_dir_path
                    )

        intrinsics_file_path = opj(local_scene_dir_path, "intrinsics.txt")
        intrinsics_url = opj(remote_scene_dir_url, "intrinsics.txt")
        if not os.path.exists(intrinsics_file_path):
            download_data(intrinsics_url, local_scene_dir_path)


# @dataclass
# class KITTIData:
#     """__getitem__ return values of KITTI class"""

#     point_cloud_coords: np.ndarray


# class KITTI:
#     def __init__(self, dataset_path: str) -> None:
#         self.dataset_path = dataset_path

#         self.velodyne_file_paths = sorted(
#             glob.glob(opj(self.dataset_path, "velodyne", "*"))
#         )

#     def __len__(self):
#         return len(self.velodyne_file_paths)

#     def __getitem__(self, idx: int) -> ScanNetData:
#         data = KITTIData()

#         data.point_cloud_coords = np.fromfile(
#             self.velodyne_file_paths[idx], dtype=np.float32
#         ).reshape(-1, 4)

#         return data
