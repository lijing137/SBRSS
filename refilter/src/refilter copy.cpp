/*
#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <algorithm>  // For std::find
#include <dirent.h>  // 用于 opendir, readdir, closedir
#include <cstring>   // 用于处理字符串（如 std::string）
#include <unordered_map>

using namespace std;

const float GRID_RESOLUTION = 0.05f; // 网格分辨率 0.05 米
const int X_MIN = -11, X_MAX = 23, Y_MIN = -7, Y_MAX = 11, Z_MIN = -1, Z_MAX = 6;

struct Point {
    float x, y, z;
};

// 网格结构体
struct Grid {
    std::vector<std::vector<std::vector<std::vector<int>>>> data;

    Grid() {
        int x_size = (X_MAX - X_MIN) / GRID_RESOLUTION + 1;
        int y_size = (Y_MAX - Y_MIN) / GRID_RESOLUTION + 1;
        int z_size = (Z_MAX - Z_MIN) / GRID_RESOLUTION + 1;
        data.resize(x_size, std::vector<std::vector<std::vector<int>>>(y_size, std::vector<std::vector<int>>(z_size)));
    }

    // 填充网格，保存多个索引值
    void fill(float x, float y, float z, int value) {
        int ix = (x - X_MIN) / GRID_RESOLUTION;
        int iy = (y - Y_MIN) / GRID_RESOLUTION;
        int iz = (z - Z_MIN) / GRID_RESOLUTION;
        if (ix >= 0 && ix < data.size() && iy >= 0 && iy < data[0].size() && iz >= 0 && iz < data[0][0].size()) {
            if (std::find(data[ix][iy][iz].begin(), data[ix][iy][iz].end(), value) == data[ix][iy][iz].end()) {
                data[ix][iy][iz].push_back(value);  // 只有当 value 不存在时才添加
            }
        }
    }

    // 获取索引值数量大于2的网格及其索引值
    std::vector<std::tuple<int, int, int, std::vector<int>>> getGridsWithMultipleValues() {
        std::vector<std::tuple<int, int, int, std::vector<int>>> result;
        for (int ix = 0; ix < data.size(); ++ix) {
            for (int iy = 0; iy < data[ix].size(); ++iy) {
                for (int iz = 0; iz < data[ix][iy].size(); ++iz) {
                    if (data[ix][iy][iz].size() > 2) {
                        result.emplace_back(ix, iy, iz, data[ix][iy][iz]);
                    }
                }
            }
        }
        return result;
    }

    // 输出存储了两个或更多值的网格单元及其索引值
    void printMultipleValuesGrids() {
        for (int ix = 0; ix < data.size(); ++ix) {
            for (int iy = 0; iy < data[ix].size(); ++iy) {
                for (int iz = 0; iz < data[ix][iy].size(); ++iz) {
                    if (data[ix][iy][iz].size() > 10) {  // 如果该网格单元包含2个或更多值
                        std::cout << "Grid (" << ix << ", " << iy << ", " << iz << ") = { ";
                        for (int val : data[ix][iy][iz]) {
                            std::cout << val << " ";
                        }
                        std::cout << "}" << std::endl;
                    }
                }
            }
        }
    }
};

// 从 CSV 文件读取并填充网格
void processCSV(const std::string &filename, Grid &grid, int value) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float x, y, z;
        char comma;
        ss >> x >> comma >> y >> comma >> z;
        grid.fill(x, y, z, value);
    }
}

std::vector<std::string> getCSVFilesInDirectory(const std::string& directoryPath) {
    std::vector<std::string> files;
    DIR* dir = opendir(directoryPath.c_str());
    
    if (dir == nullptr) {
        std::cerr << "Error: Could not open directory " << directoryPath << std::endl;
        return files;  // 如果无法打开目录，返回空的文件列表
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr) {
        // 检查文件扩展名是否为 ".csv"
        std::string filename(entry->d_name);
        if (filename.length() > 4 && filename.substr(filename.length() - 4) == ".csv") {
            // 将文件路径加入列表
            files.push_back(directoryPath + "/" + filename);
        }
    }

    closedir(dir);  // 关闭目录
    return files;
}

// 将点分类到新的点云
void classifyPoints(const std::vector<Point> &cloud, const std::vector<int> &values, const std::vector<std::tuple<int, int, int, std::vector<int>>> &grids, Grid &grid, std::unordered_map<int, std::vector<Point>> &classifiedClouds) {
    for (const Point &point : cloud) {
        int ix = (point.x - X_MIN) / GRID_RESOLUTION;
        int iy = (point.y - Y_MIN) / GRID_RESOLUTION;
        int iz = (point.z - Z_MIN) / GRID_RESOLUTION;

        if (ix >= 0 && ix < grid.data.size() && iy >= 0 && iy < grid.data[0].size() && iz >= 0 && iz < grid.data[0][0].size()) {
            // 检查是否是多值网格
            for (const auto &[gx, gy, gz, gridValues] : grids) {
                if (ix == gx && iy == gy && iz == gz) {
                    for (int value : gridValues) {
                        classifiedClouds[value].push_back(point);
                    }
                }
            }
        }
    }
}


int main() {
    Grid grid;

    std::vector<std::vector<Point>> clouds;  // 每个文件的点云
    
    // 指定包含 CSV 文件的文件夹路径
    std::string folder_path = "/home/lj/文档/Debugfilterworld";
    // 获取文件夹中的所有 CSV 文件
    std::vector<std::string> files = getCSVFilesInDirectory(folder_path);

    // std::vector<std::string> files = {"/home/lj/yf_ws/src/FAST_LIO/merged_cloud1/merged_43.csv", "/home/lj/yf_ws/src/FAST_LIO/merged_cloud1/merged_44.csv"};

    // 处理每个文件并更新网格
    for (int i = 0; i < files.size(); ++i) {
        processCSV(files[i], grid, i);
    }

    // 获取多值网格
    auto multiValueGrids = grid.getGridsWithMultipleValues();

    // 分类点
    std::unordered_map<int, std::vector<Point>> classifiedClouds;
    for (int i = 0; i < files.size(); ++i) {
        classifyPoints(clouds[i], {i + 1}, multiValueGrids, grid, classifiedClouds);
    }

    // 输出包含多个值的网格单元及其索引值
    // grid.printMultipleValuesGrids();

    // // 可选：打印非零网格位置
    // for (int i = 0; i < grid.data.size(); ++i) {
    //     for (int j = 0; j < grid.data[i].size(); ++j) {
    //         for (int k = 0; k < grid.data[i][j].size(); ++k) {
    //             if (grid.data[i][j][k] != 0) {
    //                 std::cout << "Grid (" << i << ", " << j << ", " << k << ") = " << grid.data[i][j][k] << std::endl;
    //             }
    //         }
    //     }
    // }

    return 0;
}
*/