#include "../cuda_utils.cuh"

__global__ void k_nearest_neighbors_kernel(
    int b, int c, int n, int m, int k,
    const float *__restrict__ points_coords,
    const float *__restrict__ centers_coords,
    int *__restrict__ neighbors_indices,
    float *__restrict__ neighbors_dist)
{

    int batch_index = blockIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;

    points_coords += batch_index * n * c;
    centers_coords += batch_index * m * c;
    neighbors_indices += batch_index * m * k;
    neighbors_dist += batch_index * m * k;

    for (int center_idx = index; center_idx < m; center_idx += stride)
    {
        for (int point_idx = 0; point_idx < n; ++point_idx)
        {
            // get distance
            float point_dist = 0;
            for (int c_idx = 0; c_idx < c; ++c_idx)
            {
                float c_dist = centers_coords[center_idx + c_idx * m] - points_coords[point_idx + n * c_idx];
                point_dist += (c_dist * c_dist);
            }

            // check k neighbors
            for (int knn_idx = 0; knn_idx < k; ++knn_idx)
            {
                if (neighbors_dist[center_idx * k + knn_idx] > point_dist)
                {
                    // shift k neighbors
                    for (int shift_idx = (k - 1); shift_idx > knn_idx; --shift_idx)
                    {
                        neighbors_dist[center_idx * k + shift_idx] =
                            neighbors_dist[center_idx * k + shift_idx - 1];
                        neighbors_indices[center_idx * k + shift_idx] =
                            neighbors_indices[center_idx * k + shift_idx - 1];
                    }
                    neighbors_dist[center_idx * k + knn_idx] = point_dist;
                    neighbors_indices[center_idx * k + knn_idx] = point_idx;
                    break;
                }
            }
        }
    }
}
void k_nearest_neighbors(int b, int c, int n, int m, int k,
                         const float *points_coords,
                         const float *centers_coords,
                         int *neighbors_indices,
                         float *neighbors_dist)
{
    k_nearest_neighbors_kernel<<<
        b, optimal_num_threads(n), 0, at::cuda::getCurrentCUDAStream()>>>(
        b, c, n, m, k, points_coords, centers_coords, neighbors_indices,
        neighbors_dist);
    CUDA_CHECK_ERRORS();
}
