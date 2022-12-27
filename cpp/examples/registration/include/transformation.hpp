#ifndef _TRANSFORMATION_HPP
#define _TRANSFORMATION_HPP

#include <Eigen/Dense>
#include <Eigen/Jacobi>

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

#endif
