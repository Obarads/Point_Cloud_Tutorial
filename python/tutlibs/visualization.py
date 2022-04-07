from matplotlib.pyplot import axis
import numpy as np
import k3d
from typing import List

from tutlibs.utils import color_range_rgb_to_8bit_rgb, rgb_to_hex, single_color
from tutlibs.operator import gather


class JupyterVisualizer:
    def __init__(self) -> None:
        print("Note: this class is staticmethod only.")

    @staticmethod
    def display(objects: list, camera_init: List[float] = None):
        """Visualize objects.
        Args:
            objects: object list
            camera_init: camera 9th vector.
        """
        plot = k3d.plot()

        for obj in objects:
            plot += obj

        plot.display()

        if camera_init is not None:
            plot.camera = camera_init

    @staticmethod
    def line(
        lines: np.ndarray,
        colors: np.ndarray = None,
        color_range: list = [0, 255],
        width=0.002,
        shader="simple",
    ):
        """Create line objects for visualizer.
        Args:
            lines: start and end points of lines (N, 2, 3)
            colors: RGB (N, 3) or color code (N)
            color_range: color value range for RGB (min, max color value)
            width: width of lines on visualizer

        Note:
            N: number of lines
        """
        N, _, _ = lines.shape

        # shader setup and spliting lines
        if shader == "thick":
            split_lines = np.concatenate(
                [
                    np.full((N, 1, 3), 1),
                    np.full((N, 1, 3), np.nan),
                    np.full((N, 1, 3), 1),
                ],
                axis=1,
                dtype=np.float32,
            )
            split_colors = np.full((N, 3), fill_value=1, dtype=np.uint32)
        elif shader == "simple":
            split_lines = np.concatenate(
                [
                    np.full((N, 1, 3), np.nan),
                ],
                axis=1,
                dtype=np.float32,
            )
            split_colors = np.full((N, 1), fill_value=1, dtype=np.uint32)
            # split_colors = np.tile(colors[:, np.newaxis, :], (1, 3, 1)).reshape(-1, 3)
        else:
            raise NotImplementedError()

        # xyz setup
        lines = np.concatenate([lines, split_lines], axis=1).reshape(-1, 3)

        # color setup (get color codes)
        if colors is not None:
            colors = color_range_rgb_to_8bit_rgb(colors, color_range)
            colors = rgb_to_hex(colors)
            colors = np.hstack(
                [np.tile(colors.reshape(-1, 1), (1, 2)), split_colors]
            ).reshape(-1)
        else:
            colors = []

        obj_lines = k3d.line(lines, colors=colors, width=width, shader=shader)
        return obj_lines

    @staticmethod
    def point(
        xyz: np.ndarray,
        colors: np.ndarray = None,
        color_range: List[float] = [0, 255],
        point_size: float = 0.01,
    ):
        """Create a point cloud object for visualizer.

        Args:
            xyz: XYZ positions (N, 3)
            colorts : RGB (N, 3) or color code (N)
            color_range: color value range for RGB (min, max color value)
            point_size: size of points on visualizer

        Note:
            N: number of points
        """
        # error check
        assert type(xyz) == np.ndarray
        assert type(colors) == np.ndarray or colors is None
        if colors is not None:
            assert len(colors.shape) in [
                1,
                2,
            ], "{}, Expected colors is rgb (N, 3) or color codes (N).".format(
                colors.shape
            )
            assert len(colors) == len(xyz)

        # xyz setup
        xyz = xyz.astype(np.float32)

        # color setup
        if colors is not None:
            # to 0 ~ 255 color range
            colors = color_range_rgb_to_8bit_rgb(colors, color_range)
            # to color code
            colors = rgb_to_hex(colors)
        else:
            colors = []

        obj_points = k3d.points(
            xyz, colors=colors, point_size=point_size, shader="flat"
        )
        return obj_points

    @staticmethod
    def voxel(voxels: np.ndarray, color: int = 0x0000FF):
        """Create voxel objects for visualizer.

        Args:
            voxels: voxel data, (N, N, N)
            color: hexadecimal voxel color, single color only

        Note:
            N: number of voxel on a side.
        """
        obj_voxel = k3d.voxels(voxels, color_map=(color), compression_level=1)
        # obj_voxel = k3d.sparse_voxels(voxels, [1, 1, 1], color_map=(color), compression_level=1)
        return obj_voxel

    @staticmethod
    def mesh(
        vertices: np.ndarray,
        edges: np.ndarray,
        colors: np.ndarray = None,
        color_range: List[float] = [0, 255],
    ):
        if colors is not None:
            # to 0 ~ 255 color range
            colors = color_range_rgb_to_8bit_rgb(colors, color_range)
            # to color code
            colors = rgb_to_hex(colors)
        else:
            colors = []

        obj_mesh = k3d.mesh(
            vertices=vertices, indices=edges, colors=colors, side="double"
        )
        return obj_mesh


class JupyterVisualizerUtils:
    def __init__(self) -> None:
        print("Note: this class is staticmethod only.")

    @staticmethod
    def correspondence_line(
        source_xyz, target_xyz, corr_set, line_colors: str = None
    ):
        """Create correspondence line for registration.

        Args:
            source_xyz: xyz of source points, (N, 3)
            target_xyz: xyz of target points, (M, 3)
            corr_set: indices of correspondences between source and target points (L, 2)
            line_colors: colors of correspondence lines (L, 3)
        """
        source_xyz = gather(source_xyz, corr_set[:, 0])
        target_xyz = gather(target_xyz, corr_set[:, 1])
        line_xyz = np.concatenate(
            [source_xyz[:, np.newaxis, :], target_xyz[:, np.newaxis, :]], axis=1
        )
        if line_colors is None:
            line_colors = single_color("#0000ff", len(line_xyz))

        obj_line = JupyterVisualizer.line(
            line_xyz,
            width=0.06,
            colors=line_colors,
            color_range=[0, 255],
            shader="simple",
        )
        return obj_line

    @staticmethod
    def normal_line(xyz, normals, line_colors: str = None, line_scale=0.1):
        """Create normal lines.

        Args:
            xyz: xyz of points, (N, 3)
            normals: normals, (N, 3)
            line_colors: colors of lines (N, 3)
        """
        line_xyz = np.concatenate(
            [
                xyz[:, np.newaxis, :],
                (xyz + normals * line_scale)[:, np.newaxis, :],
            ],
            axis=1,
        )

        if line_colors is None:
            line_colors = single_color("#0000ff", len(line_xyz))

        obj_line = JupyterVisualizer.line(
            line_xyz,
            width=0.06,
            colors=line_colors,
            color_range=[0, 255],
            shader="simple",
        )

        return obj_line
