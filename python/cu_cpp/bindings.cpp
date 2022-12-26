#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <pybind11/pybind11.h>

#include "knn/k_nearest_neighbors.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("k_nearest_neighbors", &k_nearest_neighbors_forward,
          "K nearest neighbors forward (CUDA)");
}
