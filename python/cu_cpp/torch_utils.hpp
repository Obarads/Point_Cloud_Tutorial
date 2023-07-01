#ifndef _UTILS_HPP
#define _UTILS_HPP

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#define CHECK_CUDA(x) \
    TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) \
    TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")

#define CHECK_IS_INT(x)                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, \
                #x " must be an int tensor")

#define CHECK_IS_FLOAT(x)                                 \
    TORCH_CHECK(x.scalar_type() == at::ScalarType::Float, \
                #x " must be a float tensor")

#define CHECK_CUDA_INT_TENSOR(x) \
    CHECK_CUDA(x);               \
    CHECK_CONTIGUOUS(x);         \
    CHECK_IS_INT(x)

#define CHECK_CUDA_FLOAT_TENSOR(x) \
    CHECK_CUDA(x);                 \
    CHECK_CONTIGUOUS(x);           \
    CHECK_IS_FLOAT(x)

// #include <string.h>
// #include <stdexcept>
// void check_tensor(at::Tensor x, const char *type_name)
// {
//     CHECK_CUDA(x);
//     CHECK_CONTIGUOUS(x);

//     // check type name
//     try
//     {
//         if (strcmp(type_name, "i") == 0)
//         {
//             CHECK_IS_INT(x);
//         }
//         else if (strcmp(type_name, "f") == 0)
//         {
//             CHECK_IS_FLOAT(x);
//         }
//         else
//         {
//             throw std::invalid_argument("type_name must be 'i' (int) or 'f' (float)");
//         }
//     }
//     catch (const std::invalid_argument &e)
//     {
//         std::cerr << "Invalid argument error: " << e.what() << '\n';
//         std::exit(1);
//     }
// }

#endif
