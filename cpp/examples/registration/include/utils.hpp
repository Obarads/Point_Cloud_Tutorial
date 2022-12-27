#ifndef _OPERATOR_HPP
#define _OPERATOR_HPP

#include <Eigen/Dense>
#include <Eigen/Jacobi>


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

#endif
