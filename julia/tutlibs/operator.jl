function square_distance(coords1, coords2)
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
    dot_product = -2 * coords1 * coords2'
    column = repeat(sum(coords1 .* coords1, dims=2), 1, size(coords2, 1))
    row = repeat(sum(coords2 .* coords2, dims=2), 1, size(coords1, 1))'
    square_dist = column + dot_product + row
    return square_dist
end

if abspath(PROGRAM_FILE) == @__FILE__
    arr1 = reshape(range(0, 23, step=1), 8, 3)
    arr2 = reshape(range(24, 47, step=1), 8, 3)
    @time square_distance(arr1, arr2)
    @time square_distance(arr1, arr2)
end