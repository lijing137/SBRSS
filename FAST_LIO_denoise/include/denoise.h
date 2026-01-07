#ifndef DENOISE_H
#define DENOISE_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>


struct ljPose6D {
    double x;
    double y;
    double z;
    double roll;
    double pitch;
    double yaw;

    // 构造函数
    ljPose6D(double x_, double y_, double z_, double roll_, double pitch_, double yaw_)
        : x(x_), y(y_), z(z_), roll(roll_), pitch(pitch_), yaw(yaw_) {}
};

// 定义 PointType 为 pcl::PointXYZI 的别名
typedef pcl::PointXYZINormal ljPointType;

// 降噪函数声明
void denoiseKeyframes(std::vector<pcl::PointCloud<ljPointType>::Ptr> keyframeLaserClouds,
                      std::vector<ljPose6D> &keyframePoses,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_filter,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_noise);

#endif  // DENOISE_H
