#include <pcl/io/ply_io.h>
#include <math.h>

#include "include/icp.hpp"


int main(int argc, char **argv)
{

    if (argc < 2)
    {
        PCL_ERROR("Please input two args.", argv[1]);
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr target_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(argv[1], *target_pcl_cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read target cloud file: ", argv[1]);
        return (-1);
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr source_pcl_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPLYFile<pcl::PointXYZ>(argv[2], *source_pcl_cloud) == -1) //* load the file
    {
        PCL_ERROR("Couldn't read source cloud file: ", argv[2]);
        return (-1);
    }

    // Load source and target data (shape: [N, 4])
    Eigen::MatrixXf source = source_pcl_cloud->getMatrixXfMap().transpose();
    Eigen::MatrixXf target = target_pcl_cloud->getMatrixXfMap().transpose();
    Eigen::Vector4f translation_vector; translation_vector << 0.0, 0.1, 0.0, 0.0;
    source = rotation(
        translation(source, translation_vector),
        "z",
        30.0 / 180.0 * M_PI
    );

    const int num_source_points = source.rows();
    Eigen::MatrixXf transformed_source;
    Eigen::MatrixXf new_transformed_source;
    Eigen::Matrix4f transformation_matrix = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f new_transformation_matrix = Eigen::Matrix4f::Identity();
    Eigen::MatrixXi correspondence_indices;
    // Eigen::MatrixXf distances(num_source_points, 1);

    PointCloud<float> target_pc;
    eigen_to_pointcloud(target_pc, target);
    kdtree *index = new kdtree(3 /*dim*/, target_pc, {10 /* max leaf */});

    int iteration = 10;
    float threshold = 0.001;
    for (int i = 0; i < iteration; i++)
    {
        transformed_source = (transformation_matrix * source.transpose()).transpose();
        correspondence_indices = brute_force_matching(index, transformed_source);
        new_transformation_matrix = estimate_transformation(transformed_source, target, correspondence_indices);

        new_transformed_source = (new_transformation_matrix * transformed_source.transpose()).transpose();
        transformation_matrix = new_transformation_matrix * transformation_matrix;

        if ((new_transformed_source - transformed_source).cwiseAbs().sum() < threshold)
        {
            break;
        }
    }

    std::cout << transformation_matrix << std::endl;
    write_matrix<Eigen::Matrix4f>(transformation_matrix, "output.txt");
}
