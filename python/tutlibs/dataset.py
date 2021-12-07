import os
from os.path import join as opj
import subprocess
import numpy as np
import json
import cv2
import scipy.io

from .io import Mesh

def download_and_unzip(www, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    zip_file = os.path.basename(www)
    if os.path.exists(zip_file):
        print("dataset download skip")
    else:
        subprocess.run(['wget', www, '--no-check-certificate'], stdout=None)

    os.system("unzip %s -d %s" % ('"'+zip_file+'"', "'"+output_path+"'"))
    os.system('rm %s' % ('"'+zip_file+'"'))

class Pix3D:
    def __init__(self, dataset_path, preprocess=True) -> None:
        self.dataset_path = dataset_path
        self.preprocess = preprocess
        
        # loading data infomation (ex: category, img_size, voxel_path ... etc.)
        with open(os.path.join(self.dataset_path, 'pix3d.json'), 'r') as f:
            self.info = json.load(f)

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        info = self.info[idx]
        img = cv2.imread(opj(self.dataset_path, info["img"]))
        mask = cv2.imread(opj(self.dataset_path, info["mask"]))
        voxel = scipy.io.loadmat(opj(self.dataset_path, info["voxel"]))["voxel"]
        obj = Mesh.read(opj(self.dataset_path, info["model"]))

        if self.preprocess:
            voxel = self.preprocess_voxel(voxel, obj)

        return img, mask, voxel, obj, info

    @staticmethod
    def preprocess_voxel(voxel, obj):
        voxel = np.transpose(voxel, (1, 2, 0))
        return voxel
