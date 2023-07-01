import torch
import numpy as np


class Transformation:
    @staticmethod
    def translation(xyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """translate xyz coordinates.

        Args:
            xyz: (B, 3, N)
            vector: (B, 3)

        Return:
            translation xyz: (B, 3, N)
        """
        return xyz + vector[:, :, None]

    @staticmethod
    def rotation(xyz: torch.Tensor, axis: str, angle: float) -> torch.Tensor:
        """Rotate xyz.

        Args:
            xyz: (N, 3)
            axis: x, y or z
            angle: radian (0 ~ 2pi)

        Return:
            rotated xyz: (N, 3)
        """
        device = xyz.device
        torch_type = xyz.dtype
        if axis == "x":
            rotation_matrix = torch.tensor(
                [
                    [1, 0, 0],
                    [0, np.cos(angle), -np.sin(angle)],
                    [0, np.sin(angle), np.cos(angle)],
                ],
                dtype=torch_type,
                device=device,
            )
        elif axis == "y":
            rotation_matrix = torch.tensor(
                [
                    [np.cos(angle), 0, np.sin(angle)],
                    [0, 1, 0],
                    [-np.sin(angle), 0, np.cos(angle)],
                ],
                dtype=torch_type,
                device=device,
            )
        elif axis == "z":
            rotation_matrix = torch.tensor(
                [
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0],
                    [0, 0, 1],
                ],
                dtype=torch_type,
                device=device,
            )
        else:
            raise ValueError()

        rotation_xyz = torch.matmul(rotation_matrix, xyz.T).T

        return rotation_xyz

    @staticmethod
    def scaling(xyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Scale xyz.

        Args:
            xyz: (N, 3)
            vector: scaling ratio (3)

        Return:
            scaled xyz: (N, 3)
        """
        return xyz * vector[None, :]
