#include "denoise.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <fstream>
#include <iomanip>  // 用于 setprecision
#include <fstream>  // 用于文件操作
#include <chrono>

using namespace std;

pcl::PointCloud<pcl::PointXYZ>::Ptr pose2pc(const std::vector<ljPose6D> vectorPose6d){
    pcl::PointCloud<pcl::PointXYZ>::Ptr res( new pcl::PointCloud<pcl::PointXYZ> ) ;
    for( auto p : vectorPose6d){
        res->points.emplace_back(p.x, p.y, p.z);
    }
    return res;
}

// lijing
float calculateDistance(const ljPose6D& a, const ljPose6D& b){
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}
// keyframePoses——所有的位置
// pointSearchIndLoop——keyframePoses指定半径内位置的索引值
// sampledKeyframesPose——最远点采样的位置
// 
// 找到最远点采样
void farthestPointSampling(const vector<ljPose6D>& keyframePoses, const vector<int>& pointSearchIndLoop, vector<ljPose6D>& sampledKeyframesPose, vector<int>& sampledIndices, int numSamples) {

    int currentKeyframeIndex = keyframePoses.size() - 1;
    sampledIndices.push_back(currentKeyframeIndex); // 将当前关键帧的索引加入采样集合

    // 设置当前关键帧的位置为 keyframePoses 的最后一行
    ljPose6D currentKeyframePose = keyframePoses.back();
    // // 打印当前关键帧位置
    // // cout << "Current Keyframe Position: " << currentKeyframe.position.transpose() << endl;

    // 根据给定的 pointSearchIndLoop，获取 nearbyKeyframesPose
    vector<ljPose6D> nearbyKeyframesPose;

    for (int index : pointSearchIndLoop) {
        if (index >= 0 && index < keyframePoses.size()) {
            nearbyKeyframesPose.push_back(keyframePoses[index]);
        }
    }
    // std::cout << "nearbyKeyframesPose size=" << nearbyKeyframesPose.size() << std::endl;

    sampledKeyframesPose.push_back(currentKeyframePose); // 第一个采样点是当前关键帧

    while (sampledIndices.size() < numSamples && sampledIndices.size() <= nearbyKeyframesPose.size() + 1) {
        float maxDistance = -1.0f;
        int farthestIndex = -1;

        // 遍历附近关键帧，找到最远的一个
        for (int i = 0; i < nearbyKeyframesPose.size(); ++i) {
            if (std::find(sampledIndices.begin(), sampledIndices.end(), pointSearchIndLoop[i]) != sampledIndices.end()) continue; // 跳过已采样的点

            // 计算当前点与采样集合中的最近点的距离
            float minDistance = std::numeric_limits<float>::max();
            for (int sampledId : sampledIndices) {
                // 计算当前点的距离
                // float distance = calculateDistance(nearbyKeyframesPose[i], (sampledId == -1 ? currentKeyframePose : nearbyKeyframesPose[sampledId]));
                float distance = calculateDistance(nearbyKeyframesPose[i], keyframePoses[sampledId]);
                minDistance = std::min(minDistance, distance);
            }

            // 如果当前点是最远的
            if (minDistance > maxDistance) {
                maxDistance = minDistance;
                farthestIndex = i; // 当前点的索引
                // std::cout << "maxDistance=" << maxDistance << "    i=" << i << std::endl;
            }
        }

        // 如果找到了最远点，添加到采样集合
        if (farthestIndex != -1) {
            sampledIndices.push_back(pointSearchIndLoop[farthestIndex]);
            sampledKeyframesPose.push_back(nearbyKeyframesPose[farthestIndex]); // 添加到采样的关键帧列表
        }
        else {
            break; // 如果没有找到新的点，退出循环
        }
    }

    // 打印采样的索引，包括当前关键帧的标识
    // cout << "Sampled Keyframe Indices: ";
    // for (int id : sampledIndices) {
    //     cout << id << " "; // 打印所有的索引，包括当前关键帧的标识
    // }
    // cout << endl;
}

void projectPointCloudToRangeImage(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_raw,
    std::vector<std::vector<std::vector<float>>>& range_image,
    std::vector<std::vector<std::vector<float>>>& inten_image,
    std::vector<std::vector<std::vector<int>>>& index_image,
    int num_rows = 32,              // 激光雷达的垂直扫描线数
    int num_cols = 2000,            // 水平方向的分辨率
    float min_angle = -16.0f,       // 最小垂直视角
    float max_angle = 15.0f         // 最大垂直视角
) {
    float vertical_fov = max_angle - min_angle;

    // 初始化range_image和inten_image，全部设置为0
    range_image.assign(num_rows, std::vector<std::vector<float>>(num_cols));
    inten_image.assign(num_rows, std::vector<std::vector<float>>(num_cols));
    index_image.assign(num_rows, std::vector<std::vector<int>>(num_cols));

    int index = 0; // 索引计数器
    // 遍历点云，将每个点投影到range_image和inten_image
    for (const auto& point : pc_raw->points) {
        // 计算距离和水平/垂直角度
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        float horiz_angle = std::atan2(point.y, point.x) * 180.0 / M_PI;  // 转换为方位角
        float vert_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI; // 转换为俯仰角

        // 计算在图像中的行列索引
        int row = static_cast<int>((vert_angle - min_angle) / vertical_fov * num_rows);
        int col = static_cast<int>((horiz_angle + 180.0f) / 360.0f * num_cols);

        // 检查行列索引是否在范围内
        if (row >= 0 && row < num_rows && col >= 0 && col < num_cols) {
            // 保存点的距离、强度和索引
            range_image[row][col].push_back(distance);
            inten_image[row][col].push_back(point.intensity);
            index_image[row][col].push_back(index);
            // if(row==21&&col==1016){
            //     std::cout << "1867point=" << point.x << "  " << point.y << "  " << point.z << std::endl;
            // }
        }
        // 增加索引
        ++index;
    }
}

// 噪声检测函数
std::vector<int> NoiseDetect(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_raw){
    // 输出调试信息
    // std::cout << "Hello, NoiseDetect!" << std::endl;
    std::vector<std::vector<std::vector<float>>> range_image;
    std::vector<std::vector<std::vector<float>>> inten_image;
    std::vector<std::vector<std::vector<int>>> index_image;

    const int rows = 32;
    const int cols = 2000;

    // 调用函数，将点云投影为range_image和inten_image
    projectPointCloudToRangeImage(pc_raw, range_image, inten_image, index_image, rows, cols, -16.0f, 15.0f);
    // saveImageToCSV(range_image, "/home/lj/lc_fastlio2_ws/src/FAST_LIO_LC/range_image.csv");
    // saveImageToCSV(inten_image, "/home/lj/lc_fastlio2_ws/src/FAST_LIO_LC/inten_image.csv");

    // 创建一个容器来保存 index_image 中的对应值
    std::vector<int> selected_index_values;
    

    // 归一化数据
    // 计算 range_image 和 inten_image 的全局最小值和最大值
    float range_min = std::numeric_limits<float>::max();
    float range_max = std::numeric_limits<float>::lowest();
    float inten_min = std::numeric_limits<float>::max();
    float inten_max = std::numeric_limits<float>::lowest();

    // 1. 计算全局最小值和最大值
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            for (size_t i = 0; i < range_image[row][col].size(); ++i) {
                // 更新range_image的最小最大值
                float range_value = range_image[row][col][i];
                if (range_value < range_min) range_min = range_value;
                if (range_value > range_max) range_max = range_value;

                // 更新inten_image的最小最大值
                float inten_value = inten_image[row][col][i];
                if (inten_value < inten_min) inten_min = inten_value;
                if (inten_value > inten_max) inten_max = inten_value;
            }
        }
    }

    // 2. 对range_image和inten_image进行归一化
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            for (size_t i = 0; i < range_image[row][col].size(); ++i) {
                // 对range_image进行归一化
                float& range_value = range_image[row][col][i];
                if (range_max > range_min) {  // 防止除以零
                    range_value = (range_value - range_min) / (range_max - range_min);
                }

                // 对inten_image进行归一化
                float& inten_value = inten_image[row][col][i];
                if (inten_max > inten_min) {  // 防止除以零
                    inten_value = (inten_value - inten_min) / (inten_max - inten_min);
                }
            }
        }
    }

    // 构造滑窗
    const std::pair<int, int> window_size = {8, 200};
    // const int samp_num = window_size.first * window_size.second;
    // 滑窗步长
    int slide_step_rows = window_size.first;
    int slide_step_cols = window_size.second;
    // 初始化协方差
    std::vector<std::vector<double>> inten_variances((rows - window_size.first) / slide_step_rows + 1, std::vector<double>((cols - window_size.second) / slide_step_cols + 1, 0.0));
    std::vector<std::vector<double>> depth_variances((rows - window_size.first) / slide_step_rows + 1, std::vector<double>((cols - window_size.second) / slide_step_cols + 1, 0.0));
    // std::cout << "Dimensions of inten_variances: " << inten_variances.size() << " x " << inten_variances[0].size() << std::endl;


    // 计算协方差
    // 因为第一个是0,所以不用<=
    for (int row = 0; row <= rows - window_size.first; row += slide_step_rows) {
        for (int col = 0; col <= cols - window_size.second; col += slide_step_cols)
        {
            std::vector<double> depth_window, inten_window;
            // std::cout << "row,col=" << row << "  " << col << std::endl;
            // 将矩阵展平方便计算方差
            for (int i = 0; i < window_size.first; ++i) {
                for (int j = 0; j < window_size.second; ++j) {
                    for (size_t k = 0; k < range_image[row+i][col+j].size(); ++k){
                        // std::cout << "range_image=" << range_image[row+i][col+j].size() << std::endl;
                        depth_window.push_back(range_image[row + i][col + j][k]);
                        inten_window.push_back(inten_image[row + i][col + j][k]);
                    }
                    // depth_window.push_back(range_image[row + i][col + j]);
                    // inten_window.push_back(inten_image[row + i][col + j]);
                }
            }
            // std::cout << "depth_window=" << depth_window.size() << std::endl;
            // std::cout << "inten_window.size()=" << inten_window.size() << std::endl;

            
            // 计算窗口内像素的平均强度
            double mean_inten = std::accumulate(inten_window.begin(), inten_window.end(), 0.0) / inten_window.size();
            double inten_variance = 0.0;
            // 计算一个窗口的强度方差
            for (const auto& inten : inten_window) {
                inten_variance += (inten - mean_inten) * (inten - mean_inten);
            }
            inten_variance /= (inten_window.size() - 1);
            
            double mean_depth = std::accumulate(depth_window.begin(), depth_window.end(), 0.0) / depth_window.size();
            double depth_variance = 0.0;
            for (const auto& depth : depth_window) {
                depth_variance += (depth - mean_depth) * (depth - mean_depth);
            }
            depth_variance /= (depth_window.size() - 1);

            // std::cout << "debugrow=" << row / slide_step_rows << "    debugcol=" << col / slide_step_cols << std::endl;
            
            inten_variances[row / slide_step_rows][col / slide_step_cols] = inten_variance;
            depth_variances[row / slide_step_rows][col / slide_step_cols] = depth_variance;
            
        }
    }

    // 阈值 #3
    float up_threshold = 0.05; //0.1,0109是0.05
    float dw_threshold = 0.0; //0.0

    // 噪声阈值
    // 计算 depth_variances 的上限和下限阈值
    std::vector<double> depth_row_max_values;
    for (const auto& row : depth_variances) {
        depth_row_max_values.push_back(*std::max_element(row.begin(), row.end()));
    }

    // double maxnum = *std::max_element(depth_row_max_values.begin(), depth_row_max_values.end());
    double depthmaxnum = *std::max_element(
    depth_row_max_values.begin(),
    depth_row_max_values.end(),
    [](double a, double b) {
        return std::isnan(a) || (!std::isnan(b) && a < b);
    });
    std::cout << "maxnum = " << depthmaxnum << std::endl;
    double up_depth_variance_threshold = up_threshold * depthmaxnum;
    double down_depth_variance_threshold = dw_threshold * depthmaxnum;
    
    // 计算 inten_variances 的上限和下限阈值
    std::vector<double> inten_row_max_values;
    for (const auto& row : inten_variances) {
        inten_row_max_values.push_back(*std::max_element(row.begin(), row.end()));
    }

    double intenmaxnum = *std::max_element(
    inten_row_max_values.begin(),
    inten_row_max_values.end(),
    [](double a, double b) {
        return std::isnan(a) || (!std::isnan(b) && a < b);
    });

    double up_inten_variance_threshold = up_threshold * intenmaxnum;
    double down_inten_variance_threshold = dw_threshold * intenmaxnum;

    std::set<std::pair<int, int>> unique_indices; // 使用集合去重复

    // 保存异常方差
    for (int row = 0; row <= rows-window_size.first; row += slide_step_rows) {        
        for (int col = 0; col <= cols-window_size.second; col += slide_step_cols) {
            // std::cout << "debugrow=" << row / slide_step_rows << "   debugcol=" << col / slide_step_cols << std::endl;
            double inten_variance = inten_variances[row / slide_step_rows][col / slide_step_cols];
            double depth_variance = depth_variances[row / slide_step_rows][col / slide_step_cols];

            // if (depth_variance >= down_depth_variance_threshold || depth_variance <= up_depth_variance_threshold ||
            //     inten_variance >= down_inten_variance_threshold || inten_variance <= up_inten_variance_threshold) {
            if (inten_variance >= down_inten_variance_threshold || inten_variance <= up_inten_variance_threshold) {
                for (int i = 0; i < window_size.first; ++i) {
                    for (int j = 0; j < window_size.second; ++j) {
                        int r = row + i;
                        int c = col + j;
                        if(r < rows && c < cols){
                            unique_indices.insert(std::make_pair(r, c));
                        }
                        // selected_indices.emplace_back(row + i, col + j);
                        // selected_index_values.push_back(index_image[row + i][col + j]);
                    }
                }
            }
        }
    }

    // 将集合转换为 vector 以返回
    // 将make_pair转化为vector
    std::vector<std::pair<int, int>> selected_indices(unique_indices.begin(), unique_indices.end());
    // savePairVectorToCSV(selected_indices, "selected_indices.csv");
    
    // 遍历 selected_indices 中的每个 (row, col) 索引
    for (const auto& index_pair : selected_indices) {
        int row = index_pair.first;
        int col = index_pair.second;

        // 检查行列是否在 index_image 的范围内
        if (row >= 0 && row < index_image.size() && col >= 0 && col < index_image[0].size()) {
            for (size_t k = 0; k < range_image[row][col].size(); ++k){
                selected_index_values.push_back(index_image[row][col][k]);
            }
            //selected_index_values.push_back(index_image[row][col]);
        }
    }
    // saveImageToCSV(selected_index_values, "/home/lj/lc_fastlio2_ws/src/FAST_LIO_LC/selected_index_values.csv");

    return selected_index_values;
}

// 将两个点云合并
pcl::PointCloud<pcl::PointXYZINormal>::Ptr mergePointClouds(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc1,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc2) 
{
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr merged(new pcl::PointCloud<pcl::PointXYZINormal>());
    *merged = *pc1 + *pc2;
    return merged;
}

// 函数：保存点云到 CSV 文件
void savemergePointCloudToCSV(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // 写入 CSV 文件头部
    outFile << "x,y,z,intensity\n";

    // 写入点云数据
    for (const auto& point : cloud->points) {
        outFile << std::fixed << std::setprecision(6)  // 保留6位小数
                << point.x << "," << point.y << "," << point.z << "," << point.intensity << "\n";
    }

    outFile.close();
    // std::cout << "Saved point cloud to: " << filename << std::endl;
}


void NoiseRemoval(
    const std::vector<int>& selected_index_values,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_raw,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_ref1,
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_ref2,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_filter,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_noise
) {
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr merged_cloud = mergePointClouds(pc_ref1, pc_ref2);
    pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
    kdtree.setInputCloud(merged_cloud);

    // 初始化 pc_filter 和 pc_noise
    *pc_filter = *pc_raw; // 复制原始点云到 pc_filter
    pc_noise->clear();    // 确保 pc_noise 是空的

    // 噪声点标记
    std::vector<int> noise_indices;
    for (int index : selected_index_values) {
        if (index >= 0 && index < pc_filter->points.size()) {
            pcl::PointXYZINormal point = pc_filter->points[index];

            // Step 1: 分析点是否是噪声，例如与参考点云比较距离、强度等
            bool is_noise = false; // 假设一个标志位来判断是否为噪声

            // 搜索半径内的点
            std::vector<int> point_indices;  // 用于存储邻近点的索引
            std::vector<float> point_distances;  // 用于存储邻近点的距离
            // #1
            float radius = 0.05f; // 搜索半径，可以根据需求调整0.1
            int num_neighbors = kdtree.radiusSearch(point, radius, point_indices, point_distances);

            // 若邻域内点的数量少于某个阈值，则认为是噪声
            // #2个
            if (num_neighbors < 10) { // 假设阈值为1
                is_noise = true;
            }

            // 如果是噪声，移除并保存到 pc_noise
            if (is_noise) {
                pc_noise->points.push_back(point); // 保存到噪声点云
                noise_indices.push_back(index);
            }
        }
    }
    // 将噪声点索引降序排列
    std::sort(noise_indices.rbegin(), noise_indices.rend());

    // 从后往前删除噪声点，避免影响后续的索引
    for (int index : noise_indices) {
        pc_filter->points.erase(pc_filter->points.begin() + index);
    }

    // 更新 pc_filter 的宽度和高度
    pc_filter->width = pc_filter->points.size();
    pc_filter->height = 1;
    pc_filter->is_dense = true;

    // 更新 pc_noise 的宽度和高度
    pc_noise->width = pc_noise->points.size();
    pc_noise->height = 1;
    pc_noise->is_dense = true;
}

void saveVectorToCSV(const std::string& filename, const std::vector<int>& data) {
    std::ofstream file(filename);

    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 写入每个整数到文件，每个值占一行
    for (const int& value : data) {
        file << value << "\n";
    }

    file.close();
    // std::cout << "Data successfully saved to " << filename << std::endl;
}

void saveImageToCSV(const std::vector<int>& image, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Could not open the file!" << std::endl;
        return;
    }

    for (const auto& value : image) {
        file << value << "\n";
    }

    file.close();
    // std::cout << "Image saved to " << filename << std::endl;
}

void UnFrustum(
    const pcl::PointCloud<ljPointType>::Ptr pc_raw,
    pcl::PointCloud<ljPointType>::Ptr pc_frustum,
    int num_rows,
    int num_cols,
    float min_angle,
    float max_angle
) {
    float vertical_fov = max_angle - min_angle;
    // // 清空pc_frustum，用来存储符合条件的点
    // 检查 pc_frustum 是否为空，如果是，则初始化
    if (pc_frustum == nullptr) {
        // pc_frustum = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>());
        pc_frustum = pcl::PointCloud<ljPointType>::Ptr(new pcl::PointCloud<ljPointType>());
    }
    pc_frustum->points.clear();

    for (int index = 0; index < pc_raw->points.size(); ++index) {
        const auto& point = pc_raw->points[index];

        // 计算距离和水平/垂直角度
        float distance = std::sqrt(point.x * point.x + point.y * point.y + point.z * point.z);
        float horiz_angle = std::atan2(point.y, point.x) * 180.0 / M_PI;
        float vert_angle = std::atan2(point.z, std::sqrt(point.x * point.x + point.y * point.y)) * 180.0 / M_PI;

        // 计算在图像中的行列索引
        int row = static_cast<int>((vert_angle - min_angle) / vertical_fov * num_rows);
        int col = static_cast<int>((horiz_angle + 180.0f) / 360.0f * num_cols);

        // 检查行列索引是否在范围内
        if (row >= 0 && row < num_rows && col >= 0 && col < num_cols) {
            // 如果在范围内，将点添加到 pc_frustum 中
            pc_frustum->points.push_back(point);
        }
    }
}

void denoiseKeyframes(std::vector<pcl::PointCloud<ljPointType>::Ptr> keyframeLaserClouds,
                      std::vector<ljPose6D> &keyframePoses,
                      // pcl::PointCloud<ljPointType>::Ptr cureKeyframeQuaryCloud(new pcl::PointCloud<ljPointType>());

                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_filter,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_noise){
    // 判断执行
    
    // std::cout << "Starting noise reduction on keyframes..." << std::endl;
    std::cout << "quary scan = " << keyframePoses.size() << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloudKeyPoses3D = pose2pc(keyframePoses);

    // 构建kd-tree
    pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr kdtreeHistoryKeyPoses(new pcl::KdTreeFLANN<pcl::PointXYZ>());
    std::vector<int> pointSearchIndLoop;
    std::vector<float> pointSearchSqDisLoop;
    // 搜索范围 #5
    double historyKeyframeSearchRadius = 0.5; // outdoor 2.5 0205-value 0.5
    kdtreeHistoryKeyPoses->setInputCloud(copy_cloudKeyPoses3D);
    // copy_cloudKeyPoses3D->back()：指的是最新的关键帧位置。
    // historyKeyframeSearchRadius：搜索半径，指定了在多大范围内查找邻近的关键帧。
    // pointSearchIndLoop：用于存储找到的邻居关键帧的索引。
    // pointSearchSqDisLoop：用于存储找到的邻居关键帧到当前关键帧的平方距离。
    // 0：表示不指定最大邻居数量，意味着在该半径内找到的所有邻居都将被返回。
    kdtreeHistoryKeyPoses->radiusSearch(copy_cloudKeyPoses3D->back(), 
                                        historyKeyframeSearchRadius, 
                                        pointSearchIndLoop, 
                                        pointSearchSqDisLoop, 
                                        0);

    // lijing add
    // std::cout << "pointSearchIndLoop = " << pointSearchIndLoop.size() << std::endl;
    // saveVectorToCSV("/home/lj/yf_ws/src/FAST_LIO/DebugPOSE/pointSearchIndLoop.csv", pointSearchIndLoop);
    // 基于搜索半径选择的参考帧
    std::vector<ljPose6D> histkeyframePoses;
    // 最远点采样选择的历史参考关键帧
    std::vector<int> sampledIndices; // 现在用于存储采样的索引
    // 第二步：利用最远点采样选择帧
    // 在半径的基础上，利用最远点采样选择 #4
    farthestPointSampling(keyframePoses, pointSearchIndLoop, histkeyframePoses, sampledIndices, 5); // 采样 4 个关键帧0205-value 5

    
    // 第三步：在4个关键帧中再按照ICP得分进行筛选2个最相似的
    std::vector<std::pair<int, float>> scores;  // 存储索引和得分
    // ICP 匹配
    pcl::IterativeClosestPoint<ljPointType, ljPointType> icp;
    pcl::PointCloud<ljPointType>::Ptr cureKeyframeQuaryCloud(new pcl::PointCloud<ljPointType>());
    pcl::PointCloud<ljPointType>::Ptr tempframeQuaryCloud(new pcl::PointCloud<ljPointType>());
    // cureKeyframeQuaryCloud->clear();  // 清空旧的点云数据
    *tempframeQuaryCloud = *keyframeLaserClouds[sampledIndices[0]]; // 获取第一个索引对应的quary点云
    cureKeyframeQuaryCloud = tempframeQuaryCloud;

    for (size_t i = 1; i < sampledIndices.size();++i){

        pcl::PointCloud<ljPointType>::Ptr cureKeyframeReferCloud(new pcl::PointCloud<ljPointType>());
        cureKeyframeReferCloud->clear();  // 清空旧的点云数据
        *cureKeyframeReferCloud = *keyframeLaserClouds[sampledIndices[i]];  // 获取第一个索引对应的refer点云
        icp.setInputSource(cureKeyframeQuaryCloud);
        icp.setInputTarget(cureKeyframeReferCloud);
        pcl::PointCloud<ljPointType>::Ptr Final(new pcl::PointCloud<ljPointType>());
        icp.align(*Final);
        scores.push_back(make_pair(sampledIndices[i], icp.getFitnessScore()));
    }             

    // 排序 scores，按照得分从高到低排序
    std::sort(scores.begin(), scores.end(), [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;  // 比较的是得分，即第二个元素
    });
    std::vector<int> FinalIndex; // 现在用于存储采样的索引
    FinalIndex.push_back(scores[0].first);
    FinalIndex.push_back(scores[1].first);
    // std::cout << "FinalIndex=" << FinalIndex.size() << std::endl;
    std::cout << "first=" << scores[0].first << "    sec=" << scores[1].first << std::endl;

    // UnFrustum
    pcl::PointCloud<ljPointType>::Ptr pc_frustum(new pcl::PointCloud<ljPointType>());
    // UnFrustum(cureKeyframeQuaryCloud, pc_frustum, 32, 2000, -16.0f, 15.0f);
    
    // Noise Detection
    // std::vector<int> selected_index_values = NoiseDetect(pc_frustum);//01042031修改，这段代码必须有
    std::vector<int> selected_index_values = NoiseDetect(cureKeyframeQuaryCloud);


    auto start = std::chrono::high_resolution_clock::now(); 
    NoiseRemoval(selected_index_values, 
             cureKeyframeQuaryCloud, //这儿和pc_frustum绑定，需要同时修改cureKeyframeQuaryCloud
             keyframeLaserClouds[FinalIndex[0]], 
             keyframeLaserClouds[FinalIndex[1]], 
             pc_filter, 
             pc_noise);
    // 将pc_filter输入，输出后替换feats_down_world
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    // 打开文件并写入持续时间
    std::ofstream outFile("/home/lj/yf_ws/src/FAST_LIO/PCD/duration.txt", std::ios::app);  // 打开文件，若文件不存在则会创建
    if (outFile.is_open()) {
        outFile << duration_ms.count() << std::endl;
        outFile.close();  // 关闭文件
        // std::cout << "持续时间已保存到duration.txt文件中。" << std::endl;
    } else {
        std::cerr << "无法打开文件！" << std::endl;
    }


}