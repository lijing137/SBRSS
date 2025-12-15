// SphericalVoxelGrid.h

#ifndef SPHERICAL_VOXEL_GRID_H
#define SPHERICAL_VOXEL_GRID_H

#include <vector>
#include <unordered_map>
#include <tuple>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 计算球体半径：获取所有点中最远点到原点的距离
float CalculateRadius(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &cloud);

// 定义一个哈希函数用于std::tuple<int, int, int>
struct TupleHash {
    template <class T1, class T2, class T3>
    std::size_t operator() (const std::tuple<T1, T2, T3>& tuple) const {
        auto h1 = std::hash<T1>{}(std::get<0>(tuple));
        auto h2 = std::hash<T2>{}(std::get<1>(tuple));
        auto h3 = std::hash<T3>{}(std::get<2>(tuple));
        return h1 ^ h2 ^ h3;
    }
};

// 自定义的球体体素滤波器
void SphericalVoxelGrid_ApplyFilter(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &inputCloud, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &outputCloud);

#endif // SPHERICAL_VOXEL_GRID_H
