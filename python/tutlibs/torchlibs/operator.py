import torch


def index2points(points: torch.Tensor, indices: torch.Tensor):
    """Construct points with indices.

    Args:
        points: points (batch_size, num_dims, num_points)
        indices: point indices (batch_size, num_points) or (batch_size, num_points, k)

    Return:
       index_points : (batch_size, num_dims, num_points) or (batch_size, num_dims, num_points, k)
    """

    B, C, N = points.shape
    index_shape = indices.shape

    indices_base = (
        torch.arange(0, B, device=points.device).view(
            -1, *[1] * (len(index_shape) - 1)
        )
        * N
    )  # if len(idx_shape) = 3, .view(-1, 1, 1)
    indices = indices + indices_base
    indices = indices.view(-1)

    # (batch_size, num_dims, num_points) -> (batch_size, num_points, num_dims)
    points = points.transpose(2, 1).contiguous()

    index_points = points.view(B * N, C)[indices, :]
    index_points = index_points.view(*index_shape, C)

    # (batch_size, num_points, num_dims) -> (batch_size, num_dims, num_points, ...)
    index_points = index_points.permute(0, -1, *range(1, len(index_shape)))
    index_points = index_points.contiguous()

    return index_points


def square_distance(points_1:torch.Tensor, points_2:torch.Tensor):
    """
    Compute the square of distances between points_1 and points_2.

    Args:
        points_1 : point features (xyz etc.) [B, C, N]
        points_2 : point features (xyz etc.) [B, C, M]

    Return:
        distances : distances between points_1 and points_2 [B, N, M]
    """
    inner = -2 * torch.matmul(points_1.transpose(2, 1), points_2)
    xyz_column = torch.sum(points_2**2, dim=1, keepdim=True)
    xyz_row = torch.sum(points_1**2, dim=1, keepdim=True).transpose(2, 1)
    square_dist = xyz_column + inner + xyz_row
    return square_dist
