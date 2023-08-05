import torch
from torch.autograd import Function
from .backend import _backend
from typing import Tuple


class KNearestNeighbor(Function):
    @staticmethod
    def forward(
        ctx, center_coords: torch.Tensor, coords: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute k nearest neighbors between coords and center_coords.

        Args:
            ctx: ctx
            center_coords: xyz of center points [B, C, M]
            coords: xyz of all points [B, C, N]
            k: number of nearest neighbors

        Returns:
            idxs: top k idx between coords and center_coords [B, M, k]
            dists: top k distances between coords and center_coords [B, M, k]

        Note:
            B: batch size
            C: channel size
            M: number of center points
            N: number of all points
        """
        coords = coords.contiguous()
        center_coords = center_coords.contiguous()
        idxs, dists = _backend.k_nearest_neighbors_forward(
            coords, center_coords, k
        )
        ctx.save_for_backward(idxs, center_coords, coords)

        return idxs, dists

    @staticmethod
    def backward(ctx, grad_idxs: torch.Tensor, grad_dists: torch.Tensor):
        """Compute distance gradient between center and k nearest neighbors.

        Args:
            grad_idxs: input gradient of k nearest neighbors [B, M, k]
            grad_dists: input gradient of distances between center and 
                k nearest neighbors [B, M, k]

        Returns:
            grad_center_coords: gradient of center point coordinates [B, C, M]
            grad_coords: gradient of all point coordinates [B, C, N]
            None: None

        Note:
            B: batch size
            C: channel size
            M: number of center points
            N: number of all points
        """
        idxs, center_coords, coords = ctx.saved_tensors
        grad_center_coords, grad_coords = _backend.k_nearest_neighbors_backward(
            grad_dists.contiguous(),
            idxs,
            center_coords,
            coords,
        )
        print("backward:", grad_center_coords)
        return grad_center_coords, grad_coords, None


k_nearest_neighbor = KNearestNeighbor.apply
