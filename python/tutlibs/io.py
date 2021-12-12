from numpy.lib.twodim_base import tri
from plyfile import PlyData, PlyElement
import numpy as np
from typing import Dict, Tuple
import open3d as o3d
import os
import urllib.request
import tarfile
import shutil

from .utils import color_range_rgb_to_8bit_rgb

#################
### ply file ###
#################

class Mesh:
    @staticmethod
    def read(file_path:str)->Tuple[np.ndarray, np.ndarray, dict]:
        """read a triangle mesh file.
        
        Args:
            file_path: a triangle mesh file (support: obj, ply)
        
        Return:
            vertices: vertices xyz of mesh (N, 3)
            triangles: triangle edge indices (M, 3)
            data: other data
        """
        def _obj(file_path):
            obj = o3d.io.read_triangle_mesh(file_path)
            _vertices = np.asarray(obj.vertices, dtype=np.float32)
            _triangles = np.asarray(obj.triangles, dtype=np.uint32)
            _data = None
            return _vertices, _triangles, _data

        def _ply(file_path):
            obj = o3d.io.read_triangle_mesh(file_path)
            _vertices = np.asarray(obj.vertices, dtype=np.float32)
            _triangles = np.asarray(obj.triangles, dtype=np.uint32)
            _data = None
            return _vertices, _triangles, _data

        support = {
            'obj': _obj,
            'ply': _ply,
        }
        extension = file_path.split('.')[-1]
        if extension in support:
            vertices, triangles, data = support[extension](file_path)
        else:
            raise NotImplementedError()

        return vertices, triangles, data

    @staticmethod
    def write(filename:str, vertices:np.ndarray, triangles:np.ndarray):
        # Vertex
        vertex = []
        vertex_prop = []

        vertex.extend([*vertices.T])
        vertex_prop.extend([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        ply_vertex = np.empty(len(vertices), dtype=vertex_prop)
        for i, p in enumerate(vertex_prop):
            ply_vertex[p[0]] = vertex[i]

        # triangle face
        face = []
        face_prop = []

        face.extend([triangles])
        face_prop.extend([('vertex_indices', 'i4', (3,))])

        ply_face = np.empty(len(triangles), dtype=face_prop)
        for i, p in enumerate(face_prop):
            ply_face[p[0]] = face[i]

        # write ply file
        ply = PlyData([PlyElement.describe(ply_vertex, 'vertex'),
                       PlyElement.describe(ply_face, 'face')], text=True)
        ply.write(filename)


class Points:
    @staticmethod
    def read(filename:str)->Tuple[np.ndarray, np.ndarray, dict]:
        def _ply(filename):
            plydata = PlyData.read(filename)
            ply_points = plydata['vertex']
            ply_properties = ply_points.data.dtype.names

            # XYZ
            xyz_properties = ['x', 'y', 'z']
            xyz = np.array([ply_points[c] for c in xyz_properties], dtype=np.float32).T

            # Color
            rgb_properties = ['red', 'green', 'blue']
            rgb = None
            if set(rgb_properties) <= set(ply_properties):
                rgb = np.array([ply_points[c] for c in rgb_properties], dtype=np.uint32).T

            data = {}
            for prop in ply_properties:
                if not prop in xyz_properties and not prop in rgb_properties:
                    data[prop] = ply_points[prop]
            return xyz, rgb, data

        def _pcd(filename):
            pcd = o3d.io.read_point_cloud(filename)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            rgb = np.asarray(pcd.colors, dtype=np.uint32)
            data = None
            return xyz, rgb, data

        support = {
            'ply': _ply,
            'pcd': _pcd
        }
        extension = filename.split('.')[-1]
        if extension in support:
            xyz, rgb, data = support[extension](filename)
        else:
            raise NotImplementedError("This funcation support followeing")

        return xyz, rgb, data

    @staticmethod
    def write(filename:str, xyz:np.ndarray, colors:np.ndarray=None,
              color_range:list=[0, 1],
              additional_data:Dict[str, np.ndarray]=None):
        """
        Write a point cloud into a ply file.
        """

        # Point cloud data and properties for writing
        points = []
        prop = []

        # XYZ
        points.extend([*xyz.T])
        prop.extend([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        # Color
        if colors is not None:
            # to 0 ~ 255 color range
            colors = color_range_rgb_to_8bit_rgb(colors, color_range)

            points.extend([*colors.T])
            prop.extend([('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])

        # other data
        if not additional_data is None:
            for key in additional_data:
                assert len(additional_data[key].shape) == 1
                points.append(additional_data[key])
                prop.append((key, additional_data[key].dtype.str))

        ply_data = np.empty(len(xyz), dtype=prop)
        for i, p in enumerate(prop):
            ply_data[p[0]] = points[i]

        ply = PlyData([PlyElement.describe(ply_data, 'vertex')], text=True)
        ply.write(filename)


if __name__ == '__main__':
    xyz, colors, _ = Points.read('../../data/pcl_data/biwi_face_database/model.pcd')
