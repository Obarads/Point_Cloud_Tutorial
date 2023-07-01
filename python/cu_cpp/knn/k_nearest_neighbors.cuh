#ifndef _K_NEAREST_NEIGHBORS_CUH
#define _K_NEAREST_NEIGHBORS_CUH

void k_nearest_neighbors(int b, int c, int n, int m, int k,
                         const float *points_coords,
                         const float *centers_coords,
                         int *neighbors_indices,
                         float *neighbors_dist);

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
                              float *grad_coords);
#endif
