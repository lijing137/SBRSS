#include "denoise.h"
#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include <iomanip>  // 用于 setprecision
#include <fstream>  // 用于文件操作
#include <chrono>
#include <unordered_map>
#include <pcl/search/kdtree.h>
#include <omp.h>
#include <pcl/common/common.h>
#include <pcl/filters/filter.h>

#include <vector>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <stdexcept>


using PointT = pcl::PointXYZINormal;
// 小工具：去 NaN/Inf（仅看 xyz），并把 is_dense 置 true
inline pcl::PointCloud<PointT>::Ptr cleanCloudXYZ(const pcl::PointCloud<PointT>::ConstPtr& in) {
    pcl::PointCloud<PointT>::Ptr out(new pcl::PointCloud<PointT>());
    out->reserve(in->size());
    out->header   = in->header;
    out->width    = 0;
    out->height   = 1;
    out->is_dense = true;

    out->points.clear();
    out->points.reserve(in->points.size());
    for (const auto& p : in->points) {
        if (pcl::isFinite(p)) { // 仅检查 x,y,z 是否有限
            out->points.push_back(p);
        }
    }
    out->width = static_cast<uint32_t>(out->points.size());
    return out;
}

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

// 新的 Cell 结构：保存统计量与索引容器
struct Cell {
    float sumR = 0.f, sumR2 = 0.f;
    float sumI = 0.f, sumI2 = 0.f;
    int   cnt  = 0;
    std::vector<int> idx; // 第二遍再 reserve 并填充
};

inline int idx2d(int r, int c, int cols){ return r*cols + c; }

// 两遍法投影：第一遍计数+统计量，第二遍仅填索引（避免大量小vector反复扩容）
void projectTwoPassWithStats(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_raw,
    std::vector<Cell>& grid, // rows*cols
    const int rows=32, const int cols=2000,
    float min_angle=-16.f, float max_angle=15.f)
{
    grid.assign(rows*cols, Cell{});
    const float vertical_fov = max_angle - min_angle;

    // --- pass 1: count + sums ---
    int index = 0;
    for (const auto& p : pc_raw->points){
        const float d = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
        const float h = std::atan2(p.y, p.x) * 180.0f / float(M_PI);
        const float v = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y)) * 180.0f / float(M_PI);
        const int r = int((v - min_angle)/vertical_fov * rows);
        const int c = int((h + 180.0f)/360.0f * cols);
        if (r>=0 && r<rows && c>=0 && c<cols){
            Cell& cell = grid[idx2d(r,c,cols)];
            cell.cnt  += 1;
            cell.sumR += d;
            cell.sumR2+= d*d;
            cell.sumI += p.intensity;
            cell.sumI2+= p.intensity*p.intensity;
        }
        ++index;
    }
    // 预留索引容量
    for (auto& cell : grid){
        if (cell.cnt>0) cell.idx.reserve(cell.cnt);
    }
    // --- pass 2: 填 index（再次计算 r,c，但不再做任何分配性的push_back膨胀） ---
    index = 0;
    for (const auto& p : pc_raw->points){
        const float h = std::atan2(p.y, p.x) * 180.0f / float(M_PI);
        const float v = std::atan2(p.z, std::sqrt(p.x*p.x + p.y*p.y)) * 180.0f / float(M_PI);
        const int r = int((v - min_angle)/ (max_angle-min_angle) * rows);
        const int c = int((h + 180.0f)/360.0f * cols);
        if (r>=0 && r<rows && c>=0 && c<cols){
            grid[idx2d(r,c,cols)].idx.push_back(index);
        }
        ++index;
    }
}

struct WindowStats { double sum=0, sum2=0; size_t n=0; };

inline double sampleVar(const WindowStats& s){
    if (s.n<=1) return 0.0;
    const double mean = s.sum / double(s.n);
    // 与原代码一致：/ (n-1)
    return (s.sum2 - double(s.n)*mean*mean) / double(s.n - 1);
}

std::vector<int> NoiseDetect_Fast(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& pc_raw){
    const int rows=32, cols=2000;
    const std::pair<int,int> win = {8,200};     // 与原值一致
    const int sr = win.first, sc = win.second;  // 步长=窗口
    const int nR = (rows - win.first)/sr + 1;
    const int nC = (cols - win.second)/sc + 1;

    // 1) 两遍法 + 统计量
    std::vector<Cell> grid;
    projectTwoPassWithStats(pc_raw, grid, rows, cols, -16.f, 15.f);

    // 2) 窗口统计 & 计算方差（强度/深度）
    std::vector<std::vector<double>> inten_var(nR, std::vector<double>(nC,0.0));
    std::vector<std::vector<double>> depth_var(nR, std::vector<double>(nC,0.0));

    double intenVarMax = 0.0, depthVarMax = 0.0;

    for (int r0=0, R=0; r0<=rows-win.first; r0+=sr, ++R){
        for (int c0=0, C=0; c0<=cols-win.second; c0+=sc, ++C){
            WindowStats iw, dw;
            for (int dr=0; dr<win.first; ++dr){
                for (int dc=0; dc<win.second; ++dc){
                    const Cell& cell = grid[idx2d(r0+dr, c0+dc, cols)];
                    if (cell.cnt==0) continue;
                    iw.sum  += cell.sumI;
                    iw.sum2 += cell.sumI2;
                    iw.n    += size_t(cell.cnt);

                    dw.sum  += cell.sumR;
                    dw.sum2 += cell.sumR2;
                    dw.n    += size_t(cell.cnt);
                }
            }
            const double iv = sampleVar(iw);
            const double dv = sampleVar(dw);
            inten_var[R][C] = iv;
            depth_var[R][C] = dv;
            if (!std::isnan(iv)) intenVarMax = std::max(intenVarMax, iv);
            if (!std::isnan(dv)) depthVarMax = std::max(depthVarMax, dv);
        }
    }

    // 3) 与原阈值逻辑保持一致（归一化已去掉，但方差与最大方差同尺度，比较结果不变）
    const double up_threshold = 0.05; // 与原值
    const double dw_threshold = 0.0;
    const double up_inten  = up_threshold * intenVarMax;
    const double down_inten= dw_threshold * intenVarMax;
    // 你现在实际上只用强度方差做筛选（深度方差被注释掉），我保持一致

    // 4) 用布尔掩码标记选中的格（代替 std::set<pair<int,int>>）
    std::vector<uint8_t> mark(rows*cols, 0);
    for (int R=0, r0=0; R<nR; ++R, r0+=sr){
        for (int C=0, c0=0; C<nC; ++C, c0+=sc){
            const double iv = inten_var[R][C];
            if (iv >= down_inten || iv <= up_inten){
                for (int dr=0; dr<win.first; ++dr){
                    for (int dc=0; dc<win.second; ++dc){
                        const int r = r0+dr, c = c0+dc;
                        if (r<rows && c<cols) mark[idx2d(r,c,cols)] = 1;
                    }
                }
            }
        }
    }

    // 5) 收集索引（无需再展开/复制任何浮点数据）
    std::vector<int> selected;
    // 这里给个保守的预留：假定 30% 的格被选中
    selected.reserve(pc_raw->points.size()/2);
    for (int r=0; r<rows; ++r){
        for (int c=0; c<cols; ++c){
            if (!mark[idx2d(r,c,cols)]) continue;
            const auto& v = grid[idx2d(r,c,cols)].idx;
            selected.insert(selected.end(), v.begin(), v.end());
        }
    }
    return selected;
}

// 计算 Otsu 阈值（针对任意二维实数矩阵）
// nbins：直方图区间数，对样本数不大时 64~128 即可
double otsu_threshold(const std::vector<std::vector<double>>& M, int nbins = 128) {
    // 收集有效值（去掉 NaN/Inf）
    std::vector<double> x;
    x.reserve(M.size() * (M.empty() ? 0 : M[0].size()));
    for (const auto& row : M) {
        for (double v : row) {
            if (std::isfinite(v)) x.push_back(v);
        }
    }
    if (x.empty()) {
        throw std::runtime_error("inten_variances has no finite values.");
    }

    // 极值
    auto [minIt, maxIt] = std::minmax_element(x.begin(), x.end());
    const double xmin = *minIt;
    const double xmax = *maxIt;

    // 全常数数据兜底：阈值 = 该常数
    if (xmin == xmax) return xmin;

    // 直方图
    nbins = std::max(2, nbins);
    const double width = (xmax - xmin) / nbins;
    std::vector<double> counts(nbins, 0.0);
    for (double v : x) {
        int idx = static_cast<int>(std::floor((v - xmin) / width));
        if (idx < 0) idx = 0;
        if (idx >= nbins) idx = nbins - 1;  // 右端点并入最后一个 bin
        counts[idx] += 1.0;
    }
    const double N = static_cast<double>(x.size());
    std::vector<double> P(nbins), centers(nbins);
    for (int i = 0; i < nbins; ++i) {
        P[i] = counts[i] / N;
        centers[i] = xmin + (i + 0.5) * width;
    }

    // 累计概率 omega 与累计加权和 mu_k
    std::vector<double> omega(nbins), mu_k(nbins);
    omega[0] = P[0];
    mu_k[0] = P[0] * centers[0];
    for (int i = 1; i < nbins; ++i) {
        omega[i] = omega[i - 1] + P[i];
        mu_k[i] = mu_k[i - 1] + P[i] * centers[i];
    }
    const double mu_T = mu_k.back();

    // 逐阈值计算类间方差，取最大
    double best_sigma = -std::numeric_limits<double>::infinity();
    int kstar = -1;
    for (int k = 0; k < nbins; ++k) {
        const double w1 = omega[k];
        const double w2 = 1.0 - w1;
        if (w1 <= 0.0 || w2 <= 0.0) continue; // 跳过端点
        const double num = mu_T * w1 - mu_k[k];      // 等价于 w1*w2*(mu2 - mu1)
        const double sigma_b2 = (num * num) / (w1 * w2);
        if (sigma_b2 > best_sigma) {
            best_sigma = sigma_b2;
            kstar = k;
        }
    }

    // 将阈值落在两个 bin 中心的中点（kstar 可能在最后一个 bin）
    if (kstar < 0) {
        // 理论上不会发生；兜底返回中位数
        std::nth_element(x.begin(), x.begin() + x.size()/2, x.end());
        return x[x.size()/2];
    }
    if (kstar < nbins - 1) {
        return 0.5 * (centers[kstar] + centers[kstar + 1]);
    } else {
        return centers[kstar];
    }
}

// 计算 Otsu 阈值：对任意二维矩阵（这里是 inten_variances）求最大类间方差阈值
double OtsuThreshold(const vector<vector<double>>& M, int nbins = 128) {
    // 1) 拉直 + 过滤非法值
    vector<double> x;
    x.reserve(M.size() * (M.empty() ? 0 : M[0].size()));
    for (const auto& row : M) {
        for (double v : row) {
            if (std::isfinite(v)) x.push_back(v);
        }
    }
    if (x.empty()) throw runtime_error("All values are NaN/Inf; cannot compute Otsu threshold.");

    // 2) 极值与常数数据兜底
    auto [min_it, max_it] = minmax_element(x.begin(), x.end());
    double xmin = *min_it, xmax = *max_it;
    if (xmin == xmax) return xmin; // 全常数：阈值=该常数

    // 3) 构建直方图（nbins个箱），并统计概率
    if (nbins < 16) nbins = 16;
    if (nbins > 512) nbins = 512;

    const double step = (xmax - xmin) / nbins;
    vector<double> counts(nbins, 0.0);
    for (double v : x) {
        // 归入对应箱，右端点并入最后一箱
        int idx = int((v - xmin) / step);
        if (idx >= nbins) idx = nbins - 1;
        if (idx < 0)     idx = 0;
        counts[idx] += 1.0;
    }

    const double N = static_cast<double>(x.size());
    vector<double> P(nbins), binCenters(nbins);
    for (int i = 0; i < nbins; ++i) {
        P[i] = counts[i] / N;
        binCenters[i] = xmin + (i + 0.5) * step;
    }

    // 4) 累计量：omega(k) 与 mu_k (累计加权和)；mu_T 为全局均值分子
    vector<double> omega(nbins, 0.0), mu_k(nbins, 0.0);
    omega[0] = P[0];
    mu_k[0]  = P[0] * binCenters[0];
    for (int i = 1; i < nbins; ++i) {
        omega[i] = omega[i - 1] + P[i];
        mu_k[i]  = mu_k[i - 1] + P[i] * binCenters[i];
    }
    const double mu_T = mu_k.back();

    // 5) 逐阈值计算类间方差 sigma_b^2(k) = ([mu_T*omega - mu_k]^2) / (omega*(1-omega))
    double best_sigma = -1.0;
    int    best_k = -1;
    for (int k = 0; k < nbins; ++k) {
        double w1 = omega[k];
        double w2 = 1.0 - w1;
        if (w1 <= 0.0 || w2 <= 0.0) continue; // 端点无效
        double num = (mu_T * w1 - mu_k[k]);
        double sigma_b2 = (num * num) / (w1 * w2);
        if (sigma_b2 > best_sigma) {
            best_sigma = sigma_b2;
            best_k = k;
        }
    }
    if (best_k < 0) {
        // 理论上不会发生；兜底返回中位数
        nth_element(x.begin(), x.begin() + x.size()/2, x.end());
        return x[x.size()/2];
    }

    // 6) 把“最佳分割位置”转回实际阈值（取相邻两个箱中心的中点；若在最后一箱，取该箱中心）
    double tau;
    if (best_k < nbins - 1) {
        tau = 0.5 * (binCenters[best_k] + binCenters[best_k + 1]);
    } else {
        tau = binCenters[best_k];
    }
    return tau;
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
    // const std::pair<int, int> window_size = {8, 200}; //hotroom 20251128
    const std::pair<int, int> window_size = {8, 200};
    // const int samp_num = window_size.first * window_size.second;
    // 滑窗步长
    int slide_step_rows = window_size.first;
    int slide_step_cols = window_size.second;
    // 初始化协方差
    std::vector<std::vector<double>> inten_variances((rows - window_size.first) / slide_step_rows + 1, std::vector<double>((cols - window_size.second) / slide_step_cols + 1, 0.0));
    std::vector<std::vector<double>> depth_variances((rows - window_size.first) / slide_step_rows + 1, std::vector<double>((cols - window_size.second) / slide_step_cols + 1, 0.0));

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
                }
            }
            
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
            
            inten_variances[row / slide_step_rows][col / slide_step_cols] = inten_variance;
            depth_variances[row / slide_step_rows][col / slide_step_cols] = depth_variance;
            
        }
    }

    // 阈值 #3
    float up_threshold = 0.9; 
    float dw_threshold = 0.2; 

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

    double tau = otsu_threshold(inten_variances, 64);
    std::cout << "tau_per = " << tau/intenmaxnum << std::endl;
    // double tau2 = OtsuThreshold(inten_variances, /*nbins=*/64);
    // std::cout << "OtsuThreshold = " << tau2 << std::endl;
    // 保存异常方差
    for (int row = 0; row <= rows-window_size.first; row += slide_step_rows) {        
        for (int col = 0; col <= cols-window_size.second; col += slide_step_cols) {
            double inten_variance = inten_variances[row / slide_step_rows][col / slide_step_cols];
            double depth_variance = depth_variances[row / slide_step_rows][col / slide_step_cols];
            if (inten_variance <= tau) {

                for (int i = 0; i < window_size.first; ++i) {
                        for (int j = 0; j < window_size.second; ++j) {
                            int r = row + i;
                            int c = col + j;
                            if(r < rows && c < cols){
                                unique_indices.insert(std::make_pair(r, c));
                            }

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
            
        }
    }

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
}

struct VoxelKey {
    int x,y,z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
};
struct VoxelKeyHasher {
    size_t operator()(const VoxelKey& k) const {
        // 简单高效的哈希（可换 Morton 编码）
        uint64_t h = 1469598103934665603ull;
        auto mix = [&](int v){
            uint64_t u = static_cast<uint32_t>(v);
            h ^= u; h *= 1099511628211ull;
        };
        mix(k.x); mix(k.y); mix(k.z);
        return (size_t)h;
    }
};

using CloudT = pcl::PointCloud<pcl::PointXYZINormal>;
using VecIdx = std::vector<int>;

static inline VoxelKey voxelOf(const pcl::PointXYZINormal& p, float inv_leaf){
    return { (int)std::floor(p.x * inv_leaf),
             (int)std::floor(p.y * inv_leaf),
             (int)std::floor(p.z * inv_leaf) };
}

// 构建体素索引
static inline void buildVoxelIndex(
    const CloudT::Ptr& cloud, float leaf,
    std::unordered_map<VoxelKey, VecIdx, VoxelKeyHasher>& grid)
{
    grid.clear();
    grid.reserve(cloud->points.size()*2);
    const float inv = 1.0f / leaf;
    for (int i=0;i<(int)cloud->points.size();++i){
        auto key = voxelOf(cloud->points[i], inv);
        grid[key].push_back(i);
    }
}

// 查询邻域内满足半径的参考点“个数”（与 KD 半径搜索一致）
static inline int neighborCountWithinRadius(
    const pcl::PointXYZINormal& q,
    const CloudT::Ptr& ref,
    const std::unordered_map<VoxelKey, VecIdx, VoxelKeyHasher>& grid,
    float leaf, float radius_sq)
{
    int cnt = 0;
    const float inv = 1.0f / leaf;
    VoxelKey k = voxelOf(q, inv);
    for (int dx=-1; dx<=1; ++dx)
    for (int dy=-1; dy<=1; ++dy)
    for (int dz=-1; dz<=1; ++dz){
        VoxelKey nk{ k.x+dx, k.y+dy, k.z+dz };
        auto it = grid.find(nk);
        if (it==grid.end()) continue;
        const auto& bucket = it->second;
        for (int idx : bucket){
            const auto& p = ref->points[idx];
            float dx = p.x - q.x, dy = p.y - q.y, dz = p.z - q.z;
            if (dx*dx + dy*dy + dz*dz <= radius_sq) ++cnt;
        }
    }
    return cnt;
}

void NoiseRemoval_Fast(
    const std::vector<int>& selected_index_values, // 原始输入
    const CloudT::Ptr& pc_raw,
    const CloudT::Ptr& pc_ref1,
    const CloudT::Ptr& pc_ref2,
    CloudT::Ptr& pc_filter,
    CloudT::Ptr& pc_noise)
{
    // 1) 构建参考体素网格（一次）
    CloudT::Ptr merged(new CloudT); *merged = *pc_ref1 + *pc_ref2;
    std::unordered_map<VoxelKey, VecIdx, VoxelKeyHasher> grid;
    constexpr float radius = 0.05f;          // 与原逻辑一致
    const float radius_sq = radius * radius;
    buildVoxelIndex(merged, radius, grid);

    // 2) 候选索引去重
    std::vector<int> sel; sel.reserve(selected_index_values.size());
    std::vector<uint8_t> is_sel(pc_raw->points.size(), 0);
    for (int idx: selected_index_values){
        if (idx>=0 && idx<(int)pc_raw->points.size() && !is_sel[idx]){
            is_sel[idx]=1; sel.push_back(idx);
        }
    }

    // 3) 标记噪声（等价阈值：<10 个邻居即噪声）
    std::vector<uint8_t> is_noise(pc_raw->points.size(), 0);
    is_noise.shrink_to_fit();
    for (int idx: sel){
        const auto& q = pc_raw->points[idx];
        int nn = neighborCountWithinRadius(q, merged, grid, radius, radius_sq);
        if (nn < 3) is_noise[idx] = 1; // 与原判断一致lijing
    }

    // 4) 一次性重建输出
    pc_filter.reset(new CloudT);
    pc_noise.reset(new CloudT);
    pc_filter->points.reserve(pc_raw->points.size());
    pc_noise->points.reserve(sel.size()); // 大致上限

    for (size_t i=0;i<pc_raw->points.size();++i){
        if (is_sel[i] && is_noise[i]) pc_noise->points.push_back(pc_raw->points[i]);
        else                          pc_filter->points.push_back(pc_raw->points[i]);
    }
    pc_filter->width  = pc_filter->points.size(); pc_filter->height = 1; pc_filter->is_dense = true;
    pc_noise->width   = pc_noise->points.size();  pc_noise->height  = 1; pc_noise->is_dense  = true;
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
            float radius = 0.15f; // 搜索半径，// hotroom20251128-0.05
            int num_neighbors = kdtree.radiusSearch(point, radius, point_indices, point_distances);

            // 若邻域内点的数量少于某个阈值，则认为是噪声
            // #2个
            // if (num_neighbors < 10) { // hotroom20251128
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
    // 清空pc_frustum，用来存储符合条件的点
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
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_filter,
                      pcl::PointCloud<pcl::PointXYZINormal>::Ptr &pc_noise){

    pcl::PointCloud<pcl::PointXYZ>::Ptr copy_cloudKeyPoses3D = pose2pc(keyframePoses);
    std::cout << "quary scan = " << keyframePoses.size() << std::endl;

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

    auto t0 = std::chrono::steady_clock::now();
    // 第三步：在4个关键帧中再按照ICP得分进行筛选2个最相似的
    // ICP 匹配

    // 直接用第一个采样的关键帧作为 query（无需额外拷贝）
    // pcl::PointCloud<ljPointType>::ConstPtr cureKeyframeQuaryCloud = keyframeLaserClouds[sampledIndices[0]];
    pcl::PointCloud<ljPointType>::Ptr cureKeyframeQuaryCloud = keyframeLaserClouds[sampledIndices[0]];   // 只是指针别名，不拷贝

    // scores[i-1] 对应 sampledIndices[i] 的 ICP 得分
    std::vector<std::pair<int, float>> scores(sampledIndices.size() - 1);

    // OpenMP 并行，每个线程独立一个 ICP 对象，只读输入点云，线程安全
    #pragma omp parallel for default(none) shared(sampledIndices, keyframeLaserClouds, cureKeyframeQuaryCloud, scores)
    for (int i = 1; i < static_cast<int>(sampledIndices.size()); ++i) {

        const int ref_idx = sampledIndices[i];
        pcl::PointCloud<ljPointType>::ConstPtr cureKeyframeReferCloud = keyframeLaserClouds[ref_idx];

        pcl::IterativeClosestPoint<ljPointType, ljPointType> icp_local;

        // ICP 参数显式设置为“正常精度”，不特意降精度
        icp_local.setMaximumIterations(10);             // PCL 默认也是 50，显式写出来方便你后面调
        icp_local.setTransformationEpsilon(1e-4);       // 收敛条件，足够严格
        icp_local.setEuclideanFitnessEpsilon(1e-4);     // 同上
        // icp_local.setUseReciprocalCorrespondences(true);// 最稳妥：和默认保持一致

        icp_local.setInputSource(cureKeyframeQuaryCloud);
        icp_local.setInputTarget(cureKeyframeReferCloud);

        pcl::PointCloud<ljPointType> alignedCloud;      // 只是占位，用于 align() 的输出
        icp_local.align(alignedCloud);                  // 使用单位初始位姿

        const float fitness = static_cast<float>(icp_local.getFitnessScore());

        // 每个 i 对应不同的下标，写入无数据竞争
        scores[i - 1] = std::make_pair(ref_idx, fitness);
    }

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << ms << " ms\n";

    // 排序 scores，按照得分从高到低排序
    std::sort(scores.begin(), scores.end(),
              [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
                  return a.second > b.second;  // 比较的是得分，即第二个元素
              });

    std::vector<int> FinalIndex; // 现在用于存储采样的索引
    FinalIndex.reserve(2);
    FinalIndex.push_back(scores[0].first);
    FinalIndex.push_back(scores[1].first);
    std::cout << "first=" << scores[0].first << "    sec=" << scores[1].first << std::endl;    


    // UnFrustum
    pcl::PointCloud<ljPointType>::Ptr pc_frustum(new pcl::PointCloud<ljPointType>());
    // UnFrustum(cureKeyframeQuaryCloud, pc_frustum, 32, 2000, -16.0f, 15.0f);
    
    // Noise Detection
    // std::vector<int> selected_index_values = NoiseDetect(pc_frustum);//01042031修改，这段代码必须有                
    
    std::vector<int> selected_index_values = NoiseDetect(cureKeyframeQuaryCloud);    

    auto start = std::chrono::high_resolution_clock::now();  

    NoiseRemoval_Fast(
        selected_index_values,
        cureKeyframeQuaryCloud,
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