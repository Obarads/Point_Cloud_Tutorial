from plyfile import PlyData, PlyElement
import numpy as np
from typing import Dict
import open3d as o3d
import os
import urllib.request
import tarfile
import shutil

from .utils import color_range_rgb_to_8bit_rgb

def get_bunny_mesh(bunny_mesh_file_path='Bunny.ply') -> o3d.geometry.TriangleMesh:
    """Download stanford bunny mesh.
    Ref code: https://github.com/isl-org/Open3D/blob/79011a9de0eb7ed921f2cf1767351fde894baab9/examples/python/open3d_tutorial.py#L229

    Return:
        mesh (o3d.geometry.TriangleMesh): loaded mesh data from a file.
    """
    bunny_path = bunny_mesh_file_path
    if not os.path.exists(bunny_path):
        print("downloading bunny mesh")
        url = "http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz"
        urllib.request.urlretrieve(url, bunny_path + ".tar.gz")
        print("extract bunny mesh")
        with tarfile.open(bunny_path + ".tar.gz") as tar:
            tar.extractall(path=os.path.dirname(bunny_path))
        shutil.move(
            os.path.join(
                os.path.dirname(bunny_path),
                "bunny",
                "reconstruction",
                "bun_zipper.ply",
            ),
            bunny_path,
        )
        os.remove(bunny_path + ".tar.gz")
        shutil.rmtree(os.path.join(os.path.dirname(bunny_path), "bunny"))
    mesh = o3d.io.read_triangle_mesh(bunny_path)
    mesh.compute_vertex_normals()
    return mesh

#################
### ply file ###
#################



class Points:
    @staticmethod
    def read(_filename):
        def ply(filename):
            plydata = PlyData.read(filename)
            ply_points = plydata['vertex']
            ply_properties = ply_points.data.dtype.names

            # XYZ
            xyz_properties = ['x', 'y', 'z']
            xyz = np.array([ply_points[c] for c in xyz_properties]).T

            # Color
            rgb_properties = ['red', 'green', 'blue']
            if set(rgb_properties) <= set(ply_properties):
                rgb = np.array([ply_points[c] for c in rgb_properties]).T

            data = {}
            for prop in ply_properties:
                if not prop in xyz_properties and not prop in rgb_properties:
                    data[prop] = ply_points[prop]
            return xyz, rgb, data

        def pcd(_filename):
            pcd = o3d.io.read_point_cloud(_filename)
            xyz = np.asarray(pcd.points)
            rgb = np.asarray(pcd.colors)
            data = None
            return xyz, rgb, data

        support = {
            'ply': ply,
            'pcd': pcd
        }
        extension = _filename.split('.')[-1]
        if extension in support:
            xyz, rgb, data = support[extension](_filename) 
        else:
            raise NotImplementedError()

        return xyz, rgb, data

    @staticmethod
    def write(filename, xyz:np.ndarray, colors:np.ndarray=None,
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
