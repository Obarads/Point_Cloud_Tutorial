import numpy as np
from typing import List

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
    @staticmethod
    def translation_3_to_4x4(matrix_3):
        matrix_4x4 = np.identity(4)
        matrix_4x4[:3, 3] = matrix_3
        return matrix_4x4

    @staticmethod
    def rotation_3x3_to_4x4(matrix_3x3):
        matrix_4x4 = np.identity(4)
        matrix_4x4[:3, :3] = matrix_3x3
        return matrix_4x4

    @staticmethod
    def composite_4x4(matrixes_4x4:List[np.ndarray]):
        return np.linalg.multi_dot(matrixes_4x4)

    @staticmethod
    def transformation_Nx3_with_4x4(points_Nx3, matrix_4x4):
        N, _ = points_Nx3.shape
        points_Nx4 = np.concatenate([points_Nx3, np.full((N,1), fill_value=1)], axis=1)
        points_Nx4 = np.matmul(matrix_4x4, points_Nx4.T).T
        points_Nx3 = points_Nx4[:, :3]
        return points_Nx3

    @staticmethod
    def transformation_Nx3_with_3x3_3(points_Nx3, R, T):
        points_Nx3 = np.matmul(R, points_Nx3.T).T + T
        return points_Nx3


