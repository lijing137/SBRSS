
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <dirent.h>  // 用于 opendir, readdir, closedir
#include <cstring>   // 用于处理字符串（如 std::string）
#include <filesystem>
#include <yaml-cpp/yaml.h> // 添加 yaml-cpp 头文件

int ScanNumThres;
float GRID_RESOLUTION;
int X_MIN, X_MAX, Y_MIN, Y_MAX, Z_MIN, Z_MAX;
std::string folder_path;

struct Point {
    float x, y, z;

    // 定义 operator==，用于比较两个 Point 对象是否相等
    bool operator==(const Point& other) const {
        return (x == other.x) && (y == other.y) && (z == other.z);
    }
};

struct Grid {
    std::vector<std::vector<std::vector<std::vector<int>>>> data;
    int ScanNumThres = 1;

    Grid() {
        int x_size = (X_MAX - X_MIN) / GRID_RESOLUTION + 1;
        int y_size = (Y_MAX - Y_MIN) / GRID_RESOLUTION + 1;
        int z_size = (Z_MAX - Z_MIN) / GRID_RESOLUTION + 1;
        data.resize(x_size, std::vector<std::vector<std::vector<int>>>(y_size, std::vector<std::vector<int>>(z_size)));
    }

    // 填充网格
    void fill(float x, float y, float z, int value) {
        int ix = (x - X_MIN) / GRID_RESOLUTION;
        int iy = (y - Y_MIN) / GRID_RESOLUTION;
        int iz = (z - Z_MIN) / GRID_RESOLUTION;

        if (ix >= 0 && ix < data.size() && iy >= 0 && iy < data[0].size() && iz >= 0 && iz < data[0][0].size()) {
            if (std::find(data[ix][iy][iz].begin(), data[ix][iy][iz].end(), value) == data[ix][iy][iz].end()) {
                data[ix][iy][iz].push_back(value);
            }
        }
    }

    // 获取索引值数量大于2的网格及其索引值,此处可以修改
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
        float x, y, z;
        char comma;
        ss >> x >> comma >> y >> comma >> z;
        // std::cout << "xyz=" << x << "  " << y << "  " << z << std::endl;
        cloud.push_back({x, y, z});
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
        file << point.x << "," << point.y << "," << point.z << std::endl;
    }

    file.close();  // 关闭文件流
    // std::cout << "点云已成功保存到 " << filePath << std::endl;
}

void removePoint(std::vector<Point>& FinalPC, const Point& point) {
    // 从 FinalPC 中移除匹配的 point
    auto it = std::find(FinalPC.begin(), FinalPC.end(), point);
    if (it != FinalPC.end()) {
        FinalPC.erase(it);  // 如果找到匹配的点，移除它
    }
}

// 将点分类到新的点云
void classifyPoints(const std::vector<Point> &cloud, 
                    int values, 
                    const std::vector<std::tuple<int, int, int, std::vector<int>>> &grids, 
                    Grid &grid, 
                    std::unordered_map<int, std::vector<Point>> &NormalClouds,
                    std::vector<Point> &FinalPC) {
    for (const Point &point : cloud) {
        int ix = (point.x - X_MIN) / GRID_RESOLUTION;
        int iy = (point.y - Y_MIN) / GRID_RESOLUTION;
        int iz = (point.z - Z_MIN) / GRID_RESOLUTION;

        if (ix >= 0 && ix < grid.data.size() && iy >= 0 && iy < grid.data[0].size() && iz >= 0 && iz < grid.data[0][0].size()) {
            // 检查是否是多值网格
            for (const auto &[gx, gy, gz, gridValues] : grids) {
                if (ix == gx && iy == gy && iz == gz) {
                    for (int value : gridValues) {
                        // 增加判断条件，只有当 value 等于 values 时，才会执行以下操作
                        if (value == values) {
                            NormalClouds[values].push_back(point);

                            removePoint(FinalPC, point);
                            // // 从 FinalPC 中移除匹配的 point
                            // auto it = std::find(FinalPC.begin(), FinalPC.end(), point);
                            // if (it != FinalPC.end()) {
                            //     // 如果找到了匹配的 point，移除它
                            //     FinalPC.erase(it);
                            // }
                        }

                    }
                }
            }
        }
    }
}

// 将点云数据保存到 CSV 文件
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
        outFile << "x,y,z\n";

        // 写入每个点
        for (const Point &point : points) {
            outFile << point.x << "," << point.y << "," << point.z << "\n";
        }

        outFile.close();
        std::cout << "Saved " << points.size() << " points to " << filename << std::endl;
    }
}

// // 保存单个点云（FinalNoise[i]）到 CSV 文件
// void savePointCloudToCSV(const std::vector<Point>& cloud, const std::string& filePath) {
//     std::ofstream file(filePath);  // 创建文件流对象

//     if (!file.is_open()) {  // 检查文件是否成功打开
//         std::cerr << "无法打开文件: " << filePath << std::endl;
//         return;
//     }

//     // 写入 CSV 文件头（可选）
//     // file << "x,y,z" << std::endl;

//     // 遍历点云并写入每个点的坐标
//     for (const auto& point : cloud) {
//         file << point.x << "," << point.y << "," << point.z << std::endl;
//     }

//     file.close();  // 关闭文件流
//     std::cout << "点云已成功保存到 " << filePath << std::endl;
// }

int main() {
    
    // 读取 YAML 配置文件
    YAML::Node config = YAML::LoadFile("/home/lj/temp_ws/src/refilter/config/param.yaml");

    // 获取配置参数
    int ScanNumThres = config["parameters"]["ScanNumThres"].as<int>();
    float GRID_RESOLUTION = config["parameters"]["GRID_RESOLUTION"].as<float>();

    int X_MIN = config["parameters"]["X_MIN"].as<int>();
    int X_MAX = config["parameters"]["X_MAX"].as<int>();
    int Y_MIN = config["parameters"]["Y_MIN"].as<int>();
    int Y_MAX = config["parameters"]["Y_MAX"].as<int>();
    int Z_MIN = config["parameters"]["Z_MIN"].as<int>();
    int Z_MAX = config["parameters"]["Z_MAX"].as<int>();

    std::string folder_path = config["parameters"]["PCD_file"].as<std::string>(); // 读取 PCD 文件路径


    Grid grid;
    std::vector<std::vector<Point>> clouds;  // 每个文件的点云
    std::vector<std::vector<Point>> FinalNoise;  // 每个文件的点云

    // 指定包含 CSV 文件的文件夹路径
    // std::string folder_path = "/home/lj/文档/DebugPCDworld";

    // int csvCount = countCSVFilesInDirectory(folder_path);
    // if (csvCount != -1) {
    //     // std::cout << "文件夹中 .csv 文件的数量是: " << csvCount << std::endl;
    // }
    // for (int i = 3; i < csvCount;++i){
    //     // 拼接文件路径和文件名
    //     std::string filePath = folder_path + "/pc_noise_" + std::to_string(i) + ".csv";
    //     // std::cout << "拼接后的文件路径是: " << filePath << std::endl;
    //     std::vector<Point> cloud;
    //     processCSV(filePath, cloud);
    //     // savePointCloudToCSV(cloud, "/home/lj/文档/PCTemp/temp.csv");
    //     // std::cout << "/*************************/" << std::endl;
    //     clouds.push_back(cloud);
    //     FinalNoise.push_back(cloud);
    //     // std::cout << "clouds_size=" << clouds.size() << std::endl;

    //     for (const Point &point : cloud) {
    //         grid.fill(point.x, point.y, point.z, i);  // 每个文件的索引值从0开始
    //     }

    // }

    // if(1){
    //     std::cout << "VoxValueNum=" << grid.data[3][3][3].size() << std::endl;
    // }

    // // 获取多值网格
    // auto multiValueGrids = grid.getGridsWithMultipleValues();

    // // 分类点
    // std::unordered_map<int, std::vector<Point>> NormalClouds;
    
    // for (int i = 0; i < csvCount; ++i) {
    //     classifyPoints(clouds[i], i, multiValueGrids, grid, NormalClouds, FinalNoise[i]);
    //     // std::cout << "class_size=" << classifiedClouds.size() << std::endl;
    //     std::string FinalfilePath = "/home/lj/temp_ws/src/refilter/NoisePC/Final_" + std::to_string(i) + ".csv";
    //     // 保存点云
    //     savePointCloudToCSV(FinalNoise[i], FinalfilePath);
    // }

    // // 保存分类后的点云到 CSV 文件
    // saveToCSV(NormalClouds);



    return 0;
}
