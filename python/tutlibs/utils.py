import time
import numpy as np

import torch


def grouping(points:np.ndarray, idx:np.ndarray):
    """Construct points according to indices.
    This function is similar to `gather` (EX. tf.gather).

    Args:
        points: points (N, C)
        idx: indices (N) or (N, k)
    
    Returns:
        grouped_points: points according to indices
    
    To use:
    >>> import numpy as np
    >>> points = np.arange(24).reshape(8,3)
    >>> points
    array([[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8],
        [ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17],
        [18, 19, 20],
        [21, 22, 23]])
    >>> idx = np.array([2, 4, 1, 1, 0, 3, 7])
    >>> grouping2(points, idx)
    array([[ 6,  7,  8],
        [12, 13, 14],
        [ 3,  4,  5],
        [ 3,  4,  5],
        [ 0,  1,  2],
        [ 9, 10, 11],
        [21, 22, 23]])

    >>> idx = np.array([[2, 3, 5],[1, 0, 0]])
    >>> N, k = idx.shape # k (such as number of k nearest neighbors)
    >>> grouping2(points, idx)
    array([[[ 6,  7,  8],
            [ 9, 10, 11],
            [15, 16, 17]],

        [[ 3,  4,  5],
            [ 0,  1,  2],
            [ 0,  1,  2]]])
    """

    # get original shape
    idx_shape = idx.shape
    N, C = points.shape

    # reshape to a row
    row_idx = idx.reshape(-1)

    # get grouped points
    row_points = points[row_idx]
    grouped_points = row_points.reshape(*idx_shape, C)

    return grouped_points

def unit_vector(vector:np.ndarray):
    """Compute unit vectors.

    Args:
        vector: (N, C)
    
    Return:
        unit vector: (N, C)
    """
    return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

def angle(vector_1:np.ndarray, vector_2:np.ndarray):
    """Compute an angle between two vectors.

    Args:
        vector_1: (N, C)
        vector_2: (N, C)
    
    Returns:
        angle: an angle between vector_1 and vector_2 (N)
    """

    unit_vector_1 = unit_vector(vector_1)
    unit_vector_2 = unit_vector(vector_2)

    return np.arccos(np.matmul(unit_vector_1[:, np.newaxis], unit_vector_2[:, :, np.newaxis])).squeeze(axis=(1,2))

def normalized_cross(vector_1:np.ndarray, vector_2:np.ndarray):
    """Cross product with normalization

    Args:
        vector_1: (N, C)
        vector_2: (N, C)

    Return:
        unit vector (cross product result)
    """

    unit_vectoor_1 = unit_vector(vector_1)
    unit_vectoor_2 = unit_vector(vector_2)

    return np.cross(unit_vectoor_1, unit_vectoor_2)

def square_distance(coords1:np.ndarray, coords2:np.ndarray) -> np.ndarray:
    """Compute the square distances between coords1 and coords2.

    Note: Depending on the input value and the dtype, negative values may
    be mixed in the return value.

    Args:
        coords1: coordinates (N, C)
        coords2: coordinates (M, C)

    Returns:
        square distances:
            square distances between coords1 and coords2 (N, M)
    """
    dot_product = -2*np.matmul(coords1, coords2.T)
    column = np.sum(coords1**2, axis=1, keepdims=True)
    row = np.sum(coords2**2, axis=1, keepdims=True).T
    square_dist = column + dot_product + row
    return square_dist




def color_range_rgb_to_8bit_rgb(colors:np.ndarray, color_range:list=[0, 1]):
    # Get color range of minimum and maximum
    min_color = color_range[0]
    max_color = color_range[1]

    # color range (min_color ~ max_color) to (0 ~ max_color-min_color)
    colors -= min_color
    max_color -= min_color

    # to 0 ~ 255 color range and uint32 type
    colors = colors / max_color * 255
    colors = colors.astype(np.uint32)

    return colors

def rgb_to_hex(rgb):
    hex = ((rgb[:, 0]<<16) + (rgb[:, 1]<<8) + rgb[:, 2])
    return hex

def time_watcher(previous_time=None, print_key=""):
    current_time = time.time()
    if previous_time is None:
        print('time_watcher start')
    else:
        print('{}: {}'.format(print_key, current_time - previous_time))
    return current_time

def t2n(torch_tensor:torch.Tensor) -> np.ndarray:
    """torch.Tensor to numpy.ndarray
    """
    return torch_tensor.detach().cpu().numpy()

