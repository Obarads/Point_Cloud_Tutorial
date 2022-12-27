#ifndef _ICP_HPP
#define _ICP_HPP

#include <cmath>
#include <cstdlib>
#include <Eigen/Dense>
#include <Eigen/Jacobi>
#include <nanoflann/nanoflann.hpp>

using kdtree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
    PointCloud<float>, 3 /* dim */
    >;

/**
 * @brief Estimate transformation matrix between 2 point clouds.
 * @param[in] source: Point cloud to be transformed [N, 3]
 * @param[in] target: Target point cloud [N, 3]
 * @return Transformation matrix between 2 point clouds [4, 4]
 */
Eigen::Matrix4f estimate_transformation(Eigen::MatrixXf source, Eigen::MatrixXf target, Eigen::MatrixXi corr_set)
{

    Eigen::Matrix4f trans_mat = Eigen::Matrix4f::Identity();

    Eigen::MatrixXf corr_source = source(corr_set(Eigen::placeholders::all, 0), Eigen::placeholders::all);
    Eigen::MatrixXf corr_target = target(corr_set(Eigen::placeholders::all, 1), Eigen::placeholders::all);

    corr_source = corr_source(Eigen::placeholders::all, Eigen::seqN(0, 3));
    corr_target = corr_target(Eigen::placeholders::all, Eigen::seqN(0, 3));

    Eigen::MatrixXf centroid_source = corr_source.colwise().mean();
    Eigen::MatrixXf centroid_target = corr_target.colwise().mean();

    corr_source = corr_source.rowwise() - centroid_source.row(0);
    corr_target = corr_target.rowwise() - centroid_target.row(0);

    Eigen::MatrixXf correlation_mat = corr_source.transpose() * corr_target;
    Eigen::JacobiSVD<Eigen::MatrixXf, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(correlation_mat);
    Eigen::MatrixXf u = svd.matrixU();
    Eigen::MatrixXf vh = svd.matrixV();

    Eigen::MatrixXf rotation_3x3 = vh.transpose() * u.transpose();
    Eigen::MatrixXf translation_3 = centroid_target - (rotation_3x3 * centroid_source.transpose()).transpose();

    trans_mat(Eigen::seqN(0, 3), Eigen::seqN(0, 3)) = rotation_3x3;
    trans_mat(Eigen::seqN(0, 3), 3) = translation_3.transpose();

    return trans_mat;
}


Eigen::MatrixXi brute_force_matching(kdtree *index, Eigen::MatrixXf source)
{

    size_t num_results = 1;
    std::vector<uint32_t> ret_index(num_results);
    std::vector<float> out_dist_sqr(num_results);
    const int num_source_points = source.rows();
    Eigen::MatrixXi correspondence_indices(num_source_points, 2);
    // Eigen::MatrixXf distances(eigen_source.rows());

    for (int i = 0; i < num_source_points; i++)
    {
        index->knnSearch(source.row(i).data(), num_results, &ret_index[0], &out_dist_sqr[0]);
        correspondence_indices(i, 0) = i;
        correspondence_indices(i, 1) = ret_index[0];
        // distances(i, 0) = out_dist_sqr[0];
    }

    return correspondence_indices;
}

#endif
