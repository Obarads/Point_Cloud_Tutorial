include("operator.jl")

function k_nearest_neighbor(coords1, coords2, k)
    """Compute k nearest neighbors between coords1 and coords2.

    Args:
        coords1: coordinates of centroid points (N, C)
        coords2: coordinates of all points (M, C)
        k: number of nearest neighbors

    Returns:
        idxs: indices of k nearest neighbors (N, k)
        square distances: square distance for kNN (N, k)
    """

    # batch proc.
    point_pairwise_distances = square_distance(coords1, coords2)
    # idxs = sortperm(point_pairwise_distances, dim=2)[:, 1:k]
    
    # square_dists = np.take_along_axis(point_pairwise_distances, idxs, axis=-1)

    return idxs
end

