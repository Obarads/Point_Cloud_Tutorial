import re
import numpy as np
import k3d

from tutlibs.utils import color_range_rgb_to_8bit_rgb, rgb_to_hex

#################
### Mesh ###
#################

class Mesh:
    @staticmethod
    def k3d(vertices, edges):
        plot = k3d.plot()
        plt_mesh = k3d.mesh(vertices=vertices, indices=edges,
                            color_map = k3d.colormaps.basic_color_maps.Jet,
                            color_range = [-1.1,2.01],
                            side='double'
                            )
        plot += plt_mesh
        plot.display()

class Line:
    @staticmethod
    def plot(lines:np.ndarray, colors:np.ndarray=None, color_range:list=[0, 1],
             width=0.002, plot:k3d.Plot=None):
        """
        Args:
            lines: (N, 2, 3)
        """
        # color setup
        if colors is not None:
            # to 0 ~ 255 color range
            colors = color_range_rgb_to_8bit_rgb(colors, color_range)
            # to color code
            colors = rgb_to_hex(colors)
        else:
            colors = []

        # plot
        if plot is None:
            plot = k3d.plot()
        else:
            assert type(plot) == k3d.Plot

        # for line in lines:
        #     plot += k3d.line(line, width=0.0010)

        N, _, _ = lines.shape
        split_arr = np.concatenate([
            np.full((N, 1, 3), 1),
            np.full((N, 1, 3), np.nan),
            np.full((N, 1, 3), 1)
        ], axis=1)
        lines = np.concatenate([lines, split_arr], axis=1).reshape(-1, 3)
        plot += k3d.line(lines, color=0xff0000, width=width)

        return plot


#################
### Points ###
#################

class Points:
    @staticmethod
    def k3d(xyz:np.ndarray, colors:np.ndarray=None, color_range:list=[0, 1],
            point_size:float=0.01, plot:k3d.Plot=None):
        """Visualize a point cloud or build visualize settings with k3d.

        Args:
            xyz (np.ndarray): XYZ positions (np.ndarray[N, 3])
            colorts (np.ndarray): RGB (np.ndarray[N, 3]) or color code (np.ndarray[N])
            color_range (List[float, float]): color value range for RGB (min, max color value)
            point_size (float): size of points on visualizer

        Note:
            N: number of points

        Examples:
            xyz = np.random.rand(10, 3)
            color = np.random.rand(10, 3)
            Points.k3d(xyz, color)
        """
        # error check
        assert type(xyz) == np.ndarray
        assert type(colors) == np.ndarray or colors is None
        if colors is not None:
            assert len(colors.shape) in [1, 2], '{}, Expected colors is rgb (N, 3) or color codes (N).'.format(colors.shape)
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

        # plot
        if plot is None:
            plot = k3d.plot()
        else:
            assert type(plot) == k3d.Plot

        points = k3d.points(xyz, colors=colors, point_size=point_size, shader='flat')
        plot += points
        plot.display()


