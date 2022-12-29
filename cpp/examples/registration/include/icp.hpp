#ifndef ICP_HPP
#define ICP_HPP

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <nanoflann/nanoflann.hpp>

using kdtree = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXf>;

/**
 * @brief Estimate transformation matrix between 2 point clouds.
 * @param[in] source: Point cloud to be transformed [N, 3]
 * @param[in] target: Target point cloud [N, 3]
 * @return Transformation matrix between 2 point clouds [4, 4]
 */
template <typename mat_t>
mat_t estimate_transformation(mat_t source, mat_t target, Eigen::MatrixXi corr_set)
{

    mat_t trans_mat(4, 4);
    trans_mat = mat_t::Identity(4, 4);
    mat_t corr_source = source(corr_set(Eigen::placeholders::all, 0), Eigen::seqN(0, 3));
    mat_t corr_target = target(corr_set(Eigen::placeholders::all, 1), Eigen::seqN(0, 3));

    mat_t centroid_source = corr_source.colwise().mean();
    mat_t centroid_target = corr_target.colwise().mean();

    corr_source.rowwise() -= centroid_source.row(0);
    corr_target.rowwise() -= centroid_target.row(0);

    mat_t correlation_mat = corr_source.transpose() * corr_target;

    Eigen::JacobiSVD<mat_t, Eigen::ComputeFullU | Eigen::ComputeFullV> svd(correlation_mat);
    mat_t u = svd.matrixU();
    mat_t vh = svd.matrixV().transpose();

    mat_t rotation_3x3 = vh.transpose() * u.transpose();
    mat_t translation_3 = centroid_target - (rotation_3x3 * centroid_source.transpose()).transpose();

    trans_mat(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = rotation_3x3;
    trans_mat(Eigen::seqN(0, 3), 3) = translation_3.transpose();

    return trans_mat;
}

template <typename mat_t, typename result_t>
Eigen::MatrixXi brute_force_matching(kdtree *index, mat_t source)
{
    size_t num_results = 1;
    std::vector<size_t> ret_index(num_results);
    std::vector<result_t> out_dist_sqr(num_results);
    const int num_source_points = source.rows();
    Eigen::MatrixXi correspondence_indices(num_source_points, 2);

    nanoflann::KNNResultSet<result_t> resultSet(num_results);
    resultSet.init(&ret_index[0], &out_dist_sqr[0]);

    std::vector<result_t> query_pt(3);

    for (int i = 0; i < num_source_points; i++)
    {
        // query_pt = source.row(i).data();
        for (int j = 0; j < 3; j++)
            query_pt.at(j) = source(i, j);
        index->index_->findNeighbors(resultSet, &query_pt[0]);
        correspondence_indices(i, 0) = i;
        correspondence_indices(i, 1) = ret_index[0];
    }

    return correspondence_indices;
}

#endif
