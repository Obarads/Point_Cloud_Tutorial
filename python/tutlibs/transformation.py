import numpy as np
from typing import List


class TransformationMatrix:
    def __init__(self) -> None:
        print("Note: this class is staticmethod only")

    @staticmethod
    def from_translation(vector_3: np.ndarray):
        """Convert a vector of translation to a 4x4 transformation matrix.

        Args:
            vector_3: a vector of translation, (3)

        Return:
            transformation matrix, (4, 4)
        """
        matrix_4x4 = np.identity(4)
        matrix_4x4[:3, 3] = vector_3
        return matrix_4x4

    @staticmethod
    def from_rotation(matrix_3x3: np.ndarray):
        """Convert a 3x3 rotation matrix to a 4x4 transformation matrix.

        Args:
            matrix_3x3: rotation matrix, (3, 3)

        Return:
            transformation matrix, (4, 4)
        """
        matrix_4x4 = np.identity(4)
        matrix_4x4[:3, :3] = matrix_3x3
        return matrix_4x4

    @staticmethod
    def from_scaling(vector_3: np.ndarray):
        """Convert a vector of scaling to a 4x4 transformation matrix.

        Args:
            vector_3: a vector of scaling, (3)

        Return:
            transformation matrix, (4, 4)
        """
        matrix_4x4 = np.identity(4)
        matrix_4x4[:, :3] *= vector_3[:, np.newaxis]
        return matrix_4x4

    @staticmethod
    def composite(matrixes_4x4: List[np.ndarray]):
        """Compute composition of transformations.

        Args:
            matrixes_4x4: list of transformation matrixes, [(4,4), ...]

        Return:
            transformation matrix, (4, 4)
        """
        return np.linalg.multi_dot(matrixes_4x4)

    @staticmethod
    def transformation(points_Nx3: np.ndarray, matrix_4x4: np.ndarray):
        """Transforme a point cloud with a transformation matrix.

        Args:
            points_Nx3: a point cloud, (N, 3)
            matrix_4x4: a transformation matrix, (4, 4)

        Return:
            point cloud, (N, 3)
        """
        N, _ = points_Nx3.shape
        points_Nx4 = np.concatenate([points_Nx3, np.full((N, 1), fill_value=1)], axis=1)
        points_Nx4 = np.matmul(matrix_4x4, points_Nx4.T).T
        points_Nx3 = points_Nx4[:, :3]
        return points_Nx3


class Transformation:
    @staticmethod
    def translation(xyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Shift xyz.
        Args:
            xyz: (N, 3)
            vector: (3)

        Return:
            translation xyz: (N, 3)
        """
        return xyz + vector[np.newaxis, :]

    @staticmethod
    def rotation(xyz: np.ndarray, axis: str, angle: float) -> np.ndarray:
        """Rotate xyz.

        Args:
            xyz: (N, 3)
            axis: x, y or z
            angle: radian (0 ~ 2pi)

        Return:
            rotated xyz: (N, 3)
        """
        if axis == "x":
            rotation_matrix = np.array(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ]
            )
        elif axis == "y":
            rotation_matrix = np.array(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ]
            )
        elif axis == "z":
            rotation_matrix = np.array(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ]
            )
        else:
            raise ValueError()

        rotation_xyz = np.matmul(rotation_matrix, xyz.T).T

        return rotation_xyz

    @staticmethod
    def scaling(xyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Scale xyz.

        Args:
            xyz: (N, 3)
            vector: scaling ratio (3)

        Return:
            scaled xyz: (N, 3)
        """
        return xyz * vector[np.newaxis, :]
