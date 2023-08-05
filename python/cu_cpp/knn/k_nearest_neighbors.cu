#include "../cuda_utils.cuh"

/**
 * @brief The function find the k nearest neighbors with brute-force search.
 *
 * The function find the k nearest neighbors with brute-force search.
 *
 * @param[in] b The number of batches.
 * @param[in] c The number of point channels.
 * @param[in] n The number of points.
 * @param[in] m The number of center points.
 * @param[in] k The number of nearest neighbors.
 * @param[in] points_coords The coordinates of the points.
 * [batch_size, channel_size, n]
 * @param[in] centers_coords The coordinates of the center points.
 * [batch_size, channel_size, m]
 * @param[in] neighbors_indices The indices of the k nearest neighbors.
 * This function expects the argument is an empty array.
 * [batch_size, num_center_points, k]
 * @param[in] neighbors_dist The distances between the center point and the k
 * nearest neighbors. This function expects the argument is an empty array.
 * [batch_size, num_center_points, k]
 */
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

/**
 * @brief The function assigns array values to the GPU for parallel processing.
 *
 * The function assigns array values to the GPU for parallel processing to
 * find the k nearest neighbors with brute-force search.
 *
 * @param[in] b The number of batches.
 * @param[in] c The number of point channels.
 * @param[in] n The number of points.
 * @param[in] m The number of center points.
 * @param[in] k The number of nearest neighbors.
 * @param[in] points_coords The coordinates of the points.
 * [batch_size, channel_size, n]
 * @param[in] centers_coords The coordinates of the center points.
 * [batch_size, channel_size, m]
 * @param[in] neighbors_indices The indices of the k nearest neighbors.
 * This function expects the argument is an empty array.
 * [batch_size, num_center_points, k]
 * @param[in] neighbors_dist The distances between the center point and the k
 * nearest neighbors. This function expects the argument is an empty array.
 * [batch_size, num_center_points, k]
 */
void k_nearest_neighbors(int b, int c, int n, int m, int k,
                         const float *points_coords,
                         const float *centers_coords,
                         int *neighbors_indices,
                         float *neighbors_dist)
{
    k_nearest_neighbors_kernel<<<
        b, optimal_num_threads(m), 0, at::cuda::getCurrentCUDAStream()>>>(
        b, c, n, m, k, points_coords, centers_coords, neighbors_indices,
        neighbors_dist);
    CUDA_CHECK_ERRORS();
}

/**
 * @brief The function calculates the gradient of the k nearest neighbors.
 *
 * The function calculates the gradient of the k nearest neighbors. The gradient
 * calculated is distance between the center point and the k nearest neighbors.
 *
 * @param[in] batch_size The number of batches.
 * @param[in] channel_size The number of point channels.
 * @param[in] num_points The number of points.
 * @param[in] num_center_points The number of center points.
 * @param[in] k The number of nearest neighbors.
 * @param[in] grad_dists The gradient of the distance between the center point
 * and the k nearest neighbors. This function expects the argument is an empty
 * array.
 * [batch_size, num_center_points, k]
 * @param[in] indices The indices of the k nearest neighbors.
 * [batch_size, channel_size, num_center_points]
 * @param[in] center_coords The coordinates of the center points.
 * [batch_size, channel_size, num_center_points]
 * @param[in] coords The coordinates of the points.
 * [batch_size, channel_size, num_points]
 * @param[in] grad_center_coords The gradient of the coordinates of the center.
 * [batch_size, channel_size, num_center_points]
 * @param[in] grad_coords The gradient of the coordinates of the points.
 * [batch_size, channel_size, num_points]
 */
__global__ void k_nearest_neighbors_grad_kernel(int batch_size,
                                                int channel_size,
                                                int num_points,
                                                int num_center_points,
                                                int k,
                                                const float *__restrict__ grad_dists,
                                                const int *__restrict__ indices,
                                                const float *__restrict__ center_coords,
                                                const float *__restrict__ coords,
                                                float *__restrict__ grad_center_coords,
                                                float *__restrict__ grad_coords)
{
    int batch_index = blockIdx.x;
    int index = threadIdx.x;
    int stride = blockDim.x;

    grad_dists += batch_index * num_center_points * k;
    grad_center_coords += batch_index * num_center_points * channel_size;
    grad_coords += batch_index * num_points * channel_size;
    indices += batch_index * num_center_points * k;
    center_coords += batch_index * num_center_points * channel_size;
    coords += batch_index * num_points * channel_size;

    for (int center_idx = index; center_idx < num_center_points; center_idx += stride)
    {
        for (int knn_idx = 0; knn_idx < k; ++knn_idx)
        {
            for (int channel_idx = 0; channel_idx < channel_size; ++channel_idx)
            {
                int indices_idx = center_idx * k + knn_idx;
                int center_coords_idx = num_center_points * channel_idx + center_idx;
                int coords_idx = num_points * channel_idx + indices[indices_idx];
                atomicAdd(grad_center_coords + center_coords_idx,
                          grad_dists[indices_idx] * (2 * center_coords[center_coords_idx] - 2 * coords[coords_idx]));
                atomicAdd(grad_coords + coords_idx,
                          grad_dists[indices_idx] * (2 * coords[coords_idx] - 2 * center_coords[center_coords_idx]));
            }
        }
    }
}

/**
 * @brief The function assigns array values to the GPU for parallel processing.
 *
 * The function assigns array values to the GPU for parallel processing to
 * compute the gradient of the k nearest neighbors.
 *
 * @param[in] batch_size The number of batches.
 * @param[in] channel_size The number of point channels.
 * @param[in] num_points The number of points.
 * @param[in] num_center_points The number of center points.
 * @param[in] k The number of nearest neighbors.
 * @param[in] grad_dists The gradient of the distance between the center point
 * and the k nearest neighbors. This function expects the argument is an empty
 * array.
 * [batch_size, num_center_points, k]
 * @param[in] indices The indices of the k nearest neighbors.
 * [batch_size, num_center_points, k]
 * @param[in] center_coords The coordinates of the center points.
 * [batch_size, channel_size, num_center_points]
 * @param[in] coords The coordinates of the points.
 * [batch_size, channel_size, num_points]
 * @param[in] grad_center_coords The gradient of the coordinates of the center.
 * [batch_size, channel_size, num_center_points]
 * @param[in] grad_coords The gradient of the coordinates of the points.
 * [batch_size, channel_size, num_points]
 */
void k_nearest_neighbors_grad(int batch_size,
                              int channel_size,
                              int num_points,
                              int num_center_points,
                              int k,
                              const float *grad_dists,
                              const int *indices,
                              const float *center_coords,
                              const float *coords,
                              float *grad_center_coords,
                              float *grad_coords)
{
    k_nearest_neighbors_grad_kernel<<<
        batch_size, optimal_num_threads(num_center_points), 0, at::cuda::getCurrentCUDAStream()>>>(
        batch_size,
        channel_size,
        num_points,
        num_center_points,
        k,
        grad_dists,
        indices,
        center_coords,
        coords,
        grad_center_coords,
        grad_coords);
    CUDA_CHECK_ERRORS();
}
