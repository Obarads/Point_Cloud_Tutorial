#include <pcl/point_types.h>
#include <pcl/features/pfh.h>
#include <pcl/io/ply_io.h>


int main(){
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ> ());
  pcl::io::loadPLYFile("../../data/bunny_pc.ply", cloud)

  pcl::PointCloud<pcl::Normal>::Ptr normal_estimation (new pcl::PointCloud<pcl::Normal> ());
  normal_estimation.setInputCloud(cloud);
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  normal_estimation.setSearchMethod (tree);
  // Output datasets
  pcl::PointCloud<pcl::Normal>::Ptr normals (new pcl::PointCloud<pcl::Normal>);

  // Use all neighbors in a sphere of radius 3cm
  normal_estimation.setRadiusSearch (0.03);

  // Compute the features
  normal_estimation.compute (*normals);

  // Create the PFH estimation class, and pass the input dataset+normals to it
  pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;
  pfh.setInputCloud (cloud);
  pfh.setInputNormals (normals);
  // alternatively, if cloud is of tpe PointNormal, do pfh.setInputNormals (cloud);

  // Create an empty kdtree representation, and pass it to the PFH estimation object.
  // Its content will be filled inside the object, based on the given input dataset (as no other search surface is given).
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ> ());
  //pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); -- older call for PCL 1.5-
  pfh.setSearchMethod (tree);

  // Output datasets
  pcl::PointCloud<pcl::PFHSignature125>::Ptr pfhs (new pcl::PointCloud<pcl::PFHSignature125> ());

  // Use all neighbors in a sphere of radius 5cm
  // IMPORTANT: the radius used here has to be larger than the radius used to estimate the surface normals!!!
  pfh.setRadiusSearch (0.05);

  // Compute the features
  pfh.compute (*pfhs);

  // pfhs->size () should have the same size as the input cloud->size ()*
}