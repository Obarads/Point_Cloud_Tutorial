#include "k_nearest_neighbors.hpp"
#include "k_nearest_neighbors.cuh"

#include "../utils.hpp"

std::vector<at::Tensor> k_nearest_neighbors_forward(at::Tensor points_coords,
                                                    at::Tensor centers_coords,
                                                    int k)
{
    CHECK_CUDA(points_coords);
    CHECK_CUDA(centers_coords);
    CHECK_CONTIGUOUS(points_coords);
    CHECK_CONTIGUOUS(centers_coords);
    CHECK_IS_FLOAT(points_coords);
    CHECK_IS_FLOAT(centers_coords);

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
