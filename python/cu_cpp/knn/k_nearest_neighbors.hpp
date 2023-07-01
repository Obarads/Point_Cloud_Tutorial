#ifndef _K_NEAREST_NEIGHBORS_HPP
#define _K_NEAREST_NEIGHBORS_HPP

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> k_nearest_neighbors_forward(at::Tensor points_coords,
                                                    at::Tensor centers_coords,
                                                    int k);
std::vector<at::Tensor> k_nearest_neighbors_backward(at::Tensor grad_dists,
                                                     at::Tensor indices,
                                                     at::Tensor centers_coords,
                                                     at::Tensor coords);
#endif
