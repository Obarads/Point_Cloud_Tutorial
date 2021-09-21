from tutlibs.utils import angle
import numpy as np

def translation(xyz:np.ndarray, vector:np.ndarray) -> np.ndarray:
    """Shift points.
    Args:
        xyz: (N, C)
        vector: (C)
    
    Return:
        translation xyz: (N, C)
    """
    return xyz + vector[np.newaxis, :]

def rotation(xyz:np.ndarray, axis:str, angle:float) -> np.ndarray:
    """Rotate points.

    Args:
        xyz: (N, C)
        axis: x, y or z
        angle: radian

    Return:
        rotated xyz: (N, C)
    """
    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError()

    rotation_xyz = np.matmul(rotation_matrix, xyz.T).T

    return rotation_xyz

def scaling(xyz:np.ndarray, ratio:float) -> np.ndarray:
    """Scale xyz.

    Args:
        xyz: (N, C)
        ratio: scaling ratio
    
    Return:
        scaled xyz: (N, C)
    """
    return xyz * ratio

class TransformationMatrix:
    def __init__(self) -> None:
        pass