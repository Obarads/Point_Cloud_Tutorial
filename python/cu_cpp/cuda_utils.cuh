#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#define MAXIMUM_THREADS_IN_BLOCK 1024

inline int optimal_num_threads(int num_processing,
                               int maximum_threads_in_block = MAXIMUM_THREADS_IN_BLOCK)
{
    const int pow_2 = std::log2(static_cast<double>(num_processing));
    return max(min(1 << pow_2, maximum_threads_in_block), 1);
}

#define CUDA_CHECK_ERRORS()                                                 \
    {                                                                       \
        cudaError_t err = cudaGetLastError();                               \
        if (cudaSuccess != err)                                             \
        {                                                                   \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
                    cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__, \
                    __FILE__);                                              \
            exit(-1);                                                       \
        }                                                                   \
    }

#endif