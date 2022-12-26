#ifndef _K_NEAREST_NEIGHBORS_CUH
#define _K_NEAREST_NEIGHBORS_CUH

void k_nearest_neighbors(int b, int c, int n, int m, int k,
                            const float *points_coords,
                            const float *centers_coords,
                            int *neighbors_indices,
                            float *neighbors_dist);
#endif
