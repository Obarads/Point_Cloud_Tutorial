#ifndef _ICP_HPP
#define _ICP_HPP

#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <nanoflann/nanoflann.hpp>
#include <Eigen/Dense>
#include <Eigen/Jacobi>

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

/**
 * @brief Rotate a point cloud.
 * @param[in] coords [N, 4]
 * @param[in] axis axis selection (x or y, z)
 * @param[in] angle radian (0 ~ 2pi)
 * @return Rotated point cloud
 */
Eigen::MatrixXf rotation(Eigen::MatrixXf coords, const char *axis, float angle)
{
    Eigen::Matrix4f rotation_matrix = Eigen::Matrix4f::Identity();
    if (axis == "x")
    {
        rotation_matrix(1, 1) = std::cos(angle);
        rotation_matrix(1, 2) = -std::sin(angle);
        rotation_matrix(2, 1) = std::sin(angle);
        rotation_matrix(2, 2) = std::cos(angle);
    }
    else if (axis == "y")
    {
        rotation_matrix(0, 0) = std::cos(angle);
        rotation_matrix(0, 2) = std::sin(angle);
        rotation_matrix(2, 0) = -std::sin(angle);
        rotation_matrix(2, 2) = std::cos(angle);
    }
    else if (axis == "z")
    {
        rotation_matrix(0, 0) = std::cos(angle);
        rotation_matrix(0, 1) = -std::sin(angle);
        rotation_matrix(1, 0) = std::sin(angle);
        rotation_matrix(1, 1) = std::cos(angle);
    }
    else
    {
        std::cerr << "axis should be x, y or z." << std::endl;
        exit(1);
    }

    Eigen::MatrixXf transformed_coords = (rotation_matrix * coords.transpose()).transpose();

    return transformed_coords;
}

/**
 * @brief Translate a point cloud.
 * @param[in] coords [N, 4]
 * @param[in] translation_vector [4]
 * @return Translated point cloud
 */
Eigen::MatrixXf translation(Eigen::MatrixXf coords, Eigen::Vector4f translation_vector)
{
    Eigen::MatrixXf transformed_coords = coords.rowwise() + translation_vector.transpose();
    return transformed_coords;
}

template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };
    using coord_t = T; //!< The type of each coordinate
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); }
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
        return false;
    }

    size_t size()
    {
        return pts.size();
    }
};

using kdtree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
    PointCloud<float>, 3 /* dim */
    >;

template <typename T>
void eigen_to_pointcloud(PointCloud<T> &points, Eigen::MatrixXf eidgen_points)
{
    int N = eidgen_points.rows();
    points.pts.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        points.pts[i].x = eidgen_points(i, 0);
        points.pts[i].y = eidgen_points(i, 1);
        points.pts[i].z = eidgen_points(i, 2);
    }
}

Eigen::MatrixXf square_distance(Eigen::MatrixXf coords_1, Eigen::MatrixXf coords_2)
{
    Eigen::MatrixXf res = -2 * coords_1 * coords_2;
    Eigen::MatrixXf res = -2 * (coords_1 * coords_2.transpose()).array();
    Eigen::MatrixXf additional_matrix(res.rows(), res.cols());
    additional_matrix = (coords_1.array() * coords_1.array()).rowwise().sum().replicate<1, res.cols()>();
    res = additional_matrix + res;
    additional_matrix = (coords_2.array() * coords_2.array()).rowwise().sum().replicate<1, res.cols()>().transpose();
    res = additional_matrix + res;
    return res;
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

template <typename T>
void write_matrix(T matrix, const char *filename)
{
    std::ofstream outputfile(filename);
    outputfile << matrix;
    outputfile.close();
}

#endif
