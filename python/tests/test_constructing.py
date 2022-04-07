import os
from os.path import join as opj
import numpy as np

from tutlibs.reconstruction import mesh_to_point, point_to_voxel
from tutlibs.io import Mesh, Points
from tutlibs.visualization import JupyterVisualizer as jv


def test_mesh_to_point(data_dir_path: str):
    vertices, triangles, data = Mesh.read(opj(data_dir_path, "bunny_tm.ply"))
    _ = mesh_to_point(vertices, triangles, 2000)
    assert True


def test_point_to_voxel(data_dir_path: str):
    coords, _, _ = Points.read(opj(data_dir_path, "bunny_pc.ply"))
    _ = point_to_voxel(coords, 0.05)
    assert True
