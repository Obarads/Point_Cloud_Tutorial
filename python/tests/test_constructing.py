import os
from os.path import join as opj
import numpy as np

from tutlibs.constructing import mesh_to_point, point_to_voxel
from tutlibs.io import Mesh, Points
from tutlibs.visualization import JupyterVisualizer as jv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_PATH = os.path.abspath(opj(BASE_DIR, "../../data"))


def test_mesh_to_point():
    vertices, triangles, data = Mesh.read(opj(DATA_DIR_PATH, "bunny_tm.ply"))
    point_cloud = mesh_to_point(vertices, triangles, 2000)
    assert True


def test_point_to_voxel():
    coords, _, _ = Points.read(opj(DATA_DIR_PATH, "bunny_pc.ply"))
    voxels = point_to_voxel(coords, 0.05)
    assert True
