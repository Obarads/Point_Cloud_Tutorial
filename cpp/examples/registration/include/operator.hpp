#ifndef _OPERATOR_HPP
#define _OPERATOR_HPP

#include <Eigen/Dense>
#include <Eigen/Jacobi>

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

#endif
