#include "k_nearest_neighbors.hpp"
#include "k_nearest_neighbors.cuh"

#include "../torch_utils.hpp"
#include <iostream>

/**
 * @brief The function find the k nearest neighbors with brute-force search.
 *
 * The function find the k nearest neighbors with brute-force search. 
 *
 * @param[in] points_coords The coordinates of the points.
 * [batch_size, channel_size, number of points]
 * @param[in] centers_coords The coordinates of the center points.
 * [batch_size, channel_size, number of center points]
 * @param[in] k The number of nearest neighbors.
 * @param[out] neighbors_indices The indices of the nearest neighbors.
 * [batch_size, number of center points, k]
 * @param[out] neighbors_dist The distances of the nearest neighbors.
 * [batch_size, number of center points, k]
 */
std::vector<at::Tensor> k_nearest_neighbors_forward(at::Tensor points_coords,
                                                    at::Tensor centers_coords,
                                                    int k)
{
    CHECK_CUDA_FLOAT_TENSOR(points_coords);
    CHECK_CUDA_FLOAT_TENSOR(centers_coords);

    int b = centers_coords.size(0);
    int c = centers_coords.size(1);
    int m = centers_coords.size(2);
    int n = points_coords.size(2);

    at::Tensor neighbors_indices = torch::zeros(
        {b, m, k},
        at::device(centers_coords.device()).dtype(at::ScalarType::Int));
    at::Tensor neighbors_dist = torch::full(
        {b, m, k}, 1024,
        at::device(centers_coords.device()).dtype(at::ScalarType::Float));

    k_nearest_neighbors(b, c, n, m, k,
                        points_coords.data_ptr<float>(),
                        centers_coords.data_ptr<float>(),
                        neighbors_indices.data_ptr<int>(),
                        neighbors_dist.data_ptr<float>());

    return {neighbors_indices, neighbors_dist};
}

/**
 * @brief The function calculates the gradient of the k nearest neighbors.
 *
 * The function calculates the gradient of the k nearest neighbors. The gradient
 * calculated is distance between the center point and the k nearest neighbors.
 *
 * @param[in] grad_dists The input gradient of distances.
 * [batch_size, number of center points, k]
 * @param[in] indices The indices of the nearest neighbors.
 * [batch_size, number of center points, k]
 * @param[in] centers_coords The coordinates of the center points.
 * [batch_size, channel_size, number of center points]
 * @param[in] coords The coordinates of the points.
 * [batch_size, channel_size, number of points]
 * @param[out] grad_center_coords The gradient of the center coordinates.
 * [batch_size, channel_size, number of center points]
 * @param[out] grad_coords The gradient of the points coordinates.
 * [batch_size, channel_size, number of points]
 */
std::vector<at::Tensor> k_nearest_neighbors_backward(at::Tensor grad_dists,
                                                     at::Tensor indices,
                                                     at::Tensor centers_coords,
                                                     at::Tensor coords)
{
    CHECK_CUDA_FLOAT_TENSOR(grad_dists);
    CHECK_CUDA_INT_TENSOR(indices);
    CHECK_CUDA_FLOAT_TENSOR(centers_coords);
    CHECK_CUDA_FLOAT_TENSOR(coords);

    int num_points = coords.size(2);
    int channel_size = coords.size(1);
    int batch_size = grad_dists.size(0);
    int num_center_points = grad_dists.size(1);
    int k = grad_dists.size(2);

    at::Tensor grad_center_coords = torch::zeros(
        {batch_size, channel_size, num_center_points},
        at::device(grad_dists.device()).dtype(at::ScalarType::Float));
    at::Tensor grad_coords = torch::zeros(
        {batch_size, channel_size, num_points},
        at::device(grad_dists.device()).dtype(at::ScalarType::Float));

    k_nearest_neighbors_grad(
        batch_size,
        channel_size,
        num_points,
        num_center_points,
        k,
        grad_dists.data_ptr<float>(),
        indices.data_ptr<int>(),
        centers_coords.data_ptr<float>(),
        coords.data_ptr<float>(),
        grad_center_coords.data_ptr<float>(),
        grad_coords.data_ptr<float>());

    return {grad_center_coords, grad_coords};
}
