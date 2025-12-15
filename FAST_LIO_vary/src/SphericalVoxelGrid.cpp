// SphericalVoxelGrid.cpp

#include "SphericalVoxelGrid.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <chrono>

using namespace std;

// 自定义体素 key（替代 tuple，减少一点开销）
struct VoxelKey {
    int ix;
    int iy;
    int iz;

    bool operator==(const VoxelKey& other) const noexcept {
        return ix == other.ix && iy == other.iy && iz == other.iz;
    }
};

// 自定义 hash（简单的 FNV 混合，足够用）
struct VoxelKeyHash {
    std::size_t operator()(const VoxelKey& k) const noexcept {
        std::size_t h = 1469598103934665603ull;
        h ^= static_cast<std::size_t>(k.ix); h *= 1099511628211ull;
        h ^= static_cast<std::size_t>(k.iy); h *= 1099511628211ull;
        h ^= static_cast<std::size_t>(k.iz); h *= 1099511628211ull;
        return h;
    }
};

// 在 double 上实现一个“无调用”的 floor -> int，
// 对任意有限 double x，结果与 static_cast<int>(std::floor(x)) 一致。
static inline int fastFloorToInt(double x) noexcept
{
    int i = static_cast<int>(x);         // C++ 规定：向 0 截断
    // 对 x >= 0:  trunc(x) == floor(x)，不会进 if
    // 对 x < 0 :
    //   - 若 x 为整数，例如 -3.0，trunc(x) = -3，static_cast<double>(i) == x，不减 1
    //   - 若 x 非整数，例如 -1.2，trunc(x) = -1，-1 > -1.2 成立，--i => -2 == floor(-1.2)
    if (static_cast<double>(i) > x) {
        --i;
    }
    return i;
}

void SphericalVoxelGrid_ApplyFilter(
    const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& inputCloud,
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr& outputCloud)
{
    if (!inputCloud || inputCloud->empty()) {
        std::cout << "输入点云为空！" << std::endl;
        return;
    }

    outputCloud->clear();

    const auto& pts = inputCloud->points;
    const int N = static_cast<int>(pts.size());
    if (N == 0) {
        return;
    }

    // 每个体素的轻量统计
    struct LeafData {
        double sum_x = 0.0;
        double sum_y = 0.0;
        double sum_z = 0.0;
        std::uint32_t count = 0;

        double cx = 0.0;
        double cy = 0.0;
        double cz = 0.0;

        pcl::PointXYZINormal bestPoint;
        double bestDist2 = std::numeric_limits<double>::max();
        bool hasBest = false;
    };

    // key -> leaf 下标
    std::unordered_map<VoxelKey, int, VoxelKeyHash> key2leafIndex;
    key2leafIndex.reserve(static_cast<std::size_t>(N / 4 + 1));  // 粗略估计体素数量

    // 所有 leaf 的数据连续存放
    std::vector<LeafData> leaves;
    leaves.reserve(static_cast<std::size_t>(N / 4 + 1));

    // 每个点属于哪个 leaf（下标）
    std::vector<int> pointLeafIndex;
    pointLeafIndex.resize(N);
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------- 第一遍：计算 key + 建 leaf + 统计 ----------
    for (int i = 0; i < N; ++i)
    {
        const auto& point = pts[i];

        // 计算 max(|x|, |y|, |z|)
        const double ax = std::abs(point.x);
        const double ay = std::abs(point.y);
        const double az = std::abs(point.z);

        double maxValue = ax > ay ? ax : ay;
        if (az > maxValue) maxValue = az;

        // 和你原来的规则完全一致：5/10/15 阈值 + 0.05/0.1/0.2 + 偏移
        double voxelSize = 0.0;
        int rangeOffset = 0;

        if (maxValue < 5.0) {
            voxelSize = 0.05;
            rangeOffset = 0;
        } else if (maxValue < 10.0) {
            voxelSize = 0.1;
            rangeOffset = 1000;
        } else {
            voxelSize = 0.2;
            rangeOffset = 2000;
        }

        const int ix = fastFloorToInt(point.x / voxelSize) + rangeOffset;
        const int iy = fastFloorToInt(point.y / voxelSize) + rangeOffset;
        const int iz = fastFloorToInt(point.z / voxelSize) + rangeOffset;

        VoxelKey key{ix, iy, iz};

        // 找到 / 新建 leaf 下标
        int leafIdx;
        auto it = key2leafIndex.find(key);
        if (it == key2leafIndex.end()) {
            leafIdx = static_cast<int>(leaves.size());
            key2leafIndex.emplace(key, leafIdx);
            leaves.emplace_back();  // 默认初始化 LeafData
        } else {
            leafIdx = it->second;
        }

        pointLeafIndex[i] = leafIdx;

        // 累加统计
        LeafData& leaf = leaves[leafIdx];
        leaf.sum_x += point.x;
        leaf.sum_y += point.y;
        leaf.sum_z += point.z;
        ++leaf.count;
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration1 = t1 - t0;
    std::ofstream outFile_resize("/home/lj/spervoxel_ws/src/FAST_LIO/PCD/duration_resize.txt", std::ios::app);  // 打开文件，若文件不存在则会创建
    if (outFile_resize.is_open()) {
        outFile_resize << duration1.count() << std::endl;
        outFile_resize.close();  // 关闭文件
    } else {
        std::cerr << "无法打开文件！" << std::endl;
    }


    auto t2 = std::chrono::high_resolution_clock::now();

    // ---------- 第二步：计算每个 leaf 的质心 ----------
    for (auto& leaf : leaves)
    {
        const double inv = 1.0 / static_cast<double>(leaf.count);
        leaf.cx = leaf.sum_x * inv;
        leaf.cy = leaf.sum_y * inv;
        leaf.cz = leaf.sum_z * inv;
        leaf.bestDist2 = std::numeric_limits<double>::max();
        leaf.hasBest = false;
    }

    // ---------- 第三步：再次遍历点，用质心找最近点（不再查哈希表） ----------
    for (int i = 0; i < N; ++i)
    {
        const auto& point = pts[i];
        const int leafIdx = pointLeafIndex[i];
        LeafData& leaf = leaves[leafIdx];

        const double dx = point.x - leaf.cx;
        const double dy = point.y - leaf.cy;
        const double dz = point.z - leaf.cz;
        const double dist2 = dx * dx + dy * dy + dz * dz;

        if (!leaf.hasBest || dist2 < leaf.bestDist2)
        {
            leaf.bestDist2 = dist2;
            leaf.bestPoint = point;
            leaf.hasBest = true;
        }
    }

    // ---------- 输出结果点云 ----------
    outputCloud->points.clear();
    outputCloud->points.reserve(leaves.size());

    for (const auto& leaf : leaves)
    {
        if (leaf.hasBest) {
            outputCloud->points.push_back(leaf.bestPoint);
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration3 = t3 - t2;
    std::ofstream outFile_down("/home/lj/spervoxel_ws/src/FAST_LIO/PCD/duration_down.txt", std::ios::app);  // 打开文件，若文件不存在则会创建
    if (outFile_down.is_open()) {
        outFile_down << duration3.count() << std::endl;
        outFile_down.close();  // 关闭文件
    } else {
        std::cerr << "无法打开文件！" << std::endl;
    }

    outputCloud->width  = static_cast<uint32_t>(outputCloud->points.size());
    outputCloud->height = 1;
    outputCloud->is_dense = true;
}



