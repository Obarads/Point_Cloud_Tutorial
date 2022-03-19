using LinearAlgebra

function feature_ransac(
    source_xyz,
    target_xyz,
    source_features,
    target_features,
    ransac_n,
    iteration,
    threshold = 0.0000001,
    init_trans_mat = nothing,
)
    """
    Args:
        source_xyz: coordinates of source points, (N, 3)
        target_xyz: coordinates of target points, (M, 3)
        source_features: features of source points, (N, C)
        target_features: features of target points, (M, C)
        ransac_n: number of samples
        iteration: number of iteration
        threshold: convergence threshold
        init_trans_mat: initialinitial transformation matrix, (4, 4)

    Return:
        trans_mat: (4, 4)
    """

    if init_trans_mat === nothing
        init_trans_mat = Matrix{Float32}(I, 4, 4)

    # batch proc.
    # idxs = sortperm(point_pairwise_distances, dim=2)[:, 1:k]

    # square_dists = np.take_along_axis(point_pairwise_distances, idxs, axis=-1)


    return idxs
end

