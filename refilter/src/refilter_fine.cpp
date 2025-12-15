
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <dirent.h>  // 用于 opendir, readdir, closedir
#include <cstring>   // 用于处理字符串（如 std::string）
#include <filesystem>
#include <cmath>


const float GRID_RESOLUTION = 0.5f; // 网格分辨率 0.5 米
const int X_MIN = -10, X_MAX = 10, Y_MIN = -10, Y_MAX = 10, Z_MIN = -5, Z_MAX = 5;
const int ScanNumThres = 250; // rgba(20, 167, 57, 1)

struct Grid {
    std::vector<std::vector<std::vector<std::vector<int>>>> data;
    
    Grid() {
        int x_size = (X_MAX - X_MIN) / GRID_RESOLUTION + 1;
        int y_size = (Y_MAX - Y_MIN) / GRID_RESOLUTION + 1;
        int z_size = (Z_MAX - Z_MIN) / GRID_RESOLUTION + 1;
        data.resize(x_size, std::vector<std::vector<std::vector<int>>>(y_size, std::vector<std::vector<int>>(z_size)));
    }

    // 填充网格
    void fill(float x, float y, float z, int value) {
        int ix = std::round((x - X_MIN) / GRID_RESOLUTION);
        int iy = std::round((y - Y_MIN) / GRID_RESOLUTION);
        int iz = std::round((z - Z_MIN) / GRID_RESOLUTION);

        if (ix >= 0 && ix < data.size() && iy >= 0 && iy < data[0].size() && iz >= 0 && iz < data[0][0].size()) {
            if (std::find(data[ix][iy][iz].begin(), data[ix][iy][iz].end(), value) == data[ix][iy][iz].end()) {
                data[ix][iy][iz].push_back(value);
            }
        }
    }

    // 获取索引值数量大于阈值的网格及其索引值,此处可以修改
    std::vector<std::tuple<int, int, int, std::vector<int>>> getGridsWithMultipleValues() {
        std::vector<std::tuple<int, int, int, std::vector<int>>> result;
        for (int ix = 0; ix < data.size(); ++ix) {
            for (int iy = 0; iy < data[ix].size(); ++iy) {
                for (int iz = 0; iz < data[ix][iy].size(); ++iz) {
                    if (data[ix][iy][iz].size() > ScanNumThres) {
                        result.emplace_back(ix, iy, iz, data[ix][iy][iz]);
                    }
                }
            }
        }
        return result;
    }
};

struct Point {
    float x, y, z, intensity;

    // 定义 operator==，用于比较两个 Point 对象是否相等
    bool operator==(const Point& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }
};

// 计算文件数量
int countCSVFilesInDirectory(const std::string& folderPath) {
    DIR* dir = opendir(folderPath.c_str());
    if (dir == nullptr) {
        std::cerr << "无法打开目录: " << folderPath << std::endl;
        return -1;  // 返回 -1 表示目录打开失败
    }

    struct dirent* entry;
    int count = 0;

    // 遍历目录中的所有文件
    while ((entry = readdir(dir)) != nullptr) {
        std::string filename(entry->d_name);

        // 检查文件扩展名是否为 .csv
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".csv") {
            count++;  // 计数符合条件的文件
        }
    }

    closedir(dir);
    return count;
}

// 读取 CSV 文件
void processCSV(const std::string &filename, std::vector<Point> &cloud) {
    std::ifstream file(filename);
    std::string line;

    // 跳过第一行（通常是标题行 "x,y,z"）
    std::getline(file, line);  // 读取并丢弃第一行

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float x, y, z, intensity;
        char comma;
        ss >> x >> comma >> y >> comma >> z >> comma >> intensity;
        // std::cout << "xyz=" << x << "  " << y << "  " << z << std::endl;
        cloud.push_back({x, y, z, intensity});
    }
}

// 这个函数用于将噪声中假阳性点去除
void removePoint(std::vector<Point>& FinalPC, const Point& point) {
    // 从 FinalPC 中移除匹配的 point
    auto it = std::find(FinalPC.begin(), FinalPC.end(), point);
    if (it != FinalPC.end()) {
        FinalPC.erase(it);  // 如果找到匹配的点，移除它
    }
}

// 将点分类到新的点云
void classifyPoints(const std::vector<Point> &cloud, //第i个文件中的点云
                    int values, //文件的索引值i
                    const std::vector<std::tuple<int, int, int, std::vector<int>>> &grids, //大于阈值的网格
                    Grid &grid, 
                    std::unordered_map<int, std::vector<Point>> &NormalClouds,//正常的点云
                    std::vector<Point> &FinalPC) {//恢复过杀后的点云
    for (const Point &point : cloud) {
        int ix = std::round((point.x - X_MIN) / GRID_RESOLUTION);
        int iy = std::round((point.y - Y_MIN) / GRID_RESOLUTION);
        int iz = std::round((point.z - Z_MIN) / GRID_RESOLUTION);

        if (ix >= 0 && ix < grid.data.size() && iy >= 0 && iy < grid.data[0].size() && iz >= 0 && iz < grid.data[0][0].size()) {
            // 检查是否是多值网格
            for (const auto &[gx, gy, gz, gridValues] : grids) { //大于阈值的网格
                if (ix == gx && iy == gy && iz == gz) {
                    for (int value : gridValues) {
                        // 增加判断条件，只有当 value 等于 values 时，才会执行以下操作，帧数等于网格中的帧数
                        if (value == values) {
                            NormalClouds[values].push_back(point);

                            removePoint(FinalPC, point);

                        }

                    }
                }
            }
        }
    }
}

// 保存点云到 CSV 文件的函数
void savePointCloudToCSV(const std::vector<Point>& cloud, const std::string& filePath) {
    std::ofstream file(filePath);  // 创建文件流对象

    if (!file.is_open()) {  // 检查文件是否成功打开
        std::cerr << "无法打开文件: " << filePath << std::endl;
        return;
    }

    // 写入 CSV 文件头（可选）
    //file << "x,y,z" << std::endl;

    // 遍历点云并写入每个点的坐标
    for (const auto& point : cloud) {
        file << point.x << "," << point.y << "," << point.z << "," << point.intensity << std::endl;
    }

    file.close();  // 关闭文件流
    // std::cout << "点云已成功保存到 " << filePath << std::endl;
}

// 将正常点保存下来
void saveToCSV(const std::unordered_map<int, std::vector<Point>> &classifiedClouds) {
    std::string SAVE_PATH = "/home/lj/temp_ws/src/refilter/NormalPC/";  // 保存路径
    for (const auto &[frame, points] : classifiedClouds) {
        // 创建文件名，例如 frame2.csv
        std::string filename = SAVE_PATH + "frame" + std::to_string(frame) + ".csv";
        std::ofstream outFile(filename);

        if (!outFile.is_open()) {
            std::cerr << "Failed to open file: " << filename << std::endl;
            continue;
        }

        // 写入标题
        outFile << "x,y,z,intensity\n";

        // 写入每个点
        for (const Point &point : points) {
            outFile << point.x << "," << point.y << "," << point.z << "," << point.intensity << "\n";
        }

        outFile.close();
        // std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
    }
}

int main(){
    // 初始化网格
    Grid grid;
    std::vector<std::vector<Point>> clouds;  // 每个文件的点云
    std::vector<std::vector<Point>> FinalNoise;  // 每个文件的点云

    // 指定包含 CSV 文件的文件夹路径
    std::string folder_path = "/home/lj/temp_ws/src/refilter/Debugnoise";
    int csvCount = countCSVFilesInDirectory(folder_path);
    if (csvCount != -1) {
        // std::cout << "文件夹中 .csv 文件的数量是: " << csvCount << std::endl;
    }

    for (int i = 0; i < csvCount;++i){
        // 拼接文件路径和文件名
        std::string filePath = folder_path + "/pc_noise_" + std::to_string(i) + ".csv";
        // std::cout << "拼接后的文件路径是: " << filePath << std::endl;
        std::vector<Point> cloud;
        // 读取CSV文件，将点云保存到cloud中
        processCSV(filePath, cloud);
        // 将多个文件中的cloud保存到clouds中
        clouds.push_back(cloud);
        FinalNoise.push_back(cloud);

        for (const Point &point : cloud) {
            grid.fill(point.x, point.y, point.z, i);  // 每个文件的索引值从0开始
        }
    }
    // 测试网格中的帧数
    if(1){
        std::cout << "VoxValueNum=" << grid.data[11][9][10].size() << std::endl;
        std::cout << "VoxValueNum=" << grid.data[6][17][10].size() << std::endl;
        std::cout << "VoxValueNum=" << grid.data[8][16][10].size() << std::endl;
    }

    // 获取储存帧数的网格
    auto multiValueGrids = grid.getGridsWithMultipleValues();

    // 获取正常点
    std::unordered_map<int, std::vector<Point>> NormalClouds;

    for (int i = 0; i < csvCount; ++i) {
        classifyPoints(clouds[i], i, multiValueGrids, grid, NormalClouds, FinalNoise[i]);
        // std::cout << "class_size=" << classifiedClouds.size() << std::endl;
        std::string FinalfilePath = "/home/lj/temp_ws/src/refilter/NoisePC/Final_" + std::to_string(i) + ".csv";
        // 保存点云
        savePointCloudToCSV(FinalNoise[i], FinalfilePath);
    }

    // 保存分类后的点云到 CSV 文件
    saveToCSV(NormalClouds);

    return 0;
}