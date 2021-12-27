import os
from os.path import join as opj
import numpy as np

from tutlibs.constructing import mesh_to_point
from tutlibs.io import Mesh
from tutlibs.visualization import JupyterVisualizer as jv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR_PATH = os.path.abspath(opj(BASE_DIR, "../../data"))

def test_mesh_to_point():
    vertices, triangles, data = Mesh.read(opj(DATA_DIR_PATH, "bunny_tm.ply"))
    point_cloud = mesh_to_point(vertices, triangles, 2000)
    # obj_points = jv.point(point_cloud)
    # jv.display([obj_points])
    assert True

