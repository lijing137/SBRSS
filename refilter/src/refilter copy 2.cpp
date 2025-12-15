/*
#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_map>
#include <sstream>
#include <algorithm>
#include <dirent.h>  // 用于 opendir, readdir, closedir
#include <cstring>   // 用于处理字符串（如 std::string）
#include <filesystem>


const float GRID_RESOLUTION = 0.05f; // 网格分辨率 0.05 米
const int X_MIN = -11, X_MAX = 23, Y_MIN = -7, Y_MAX = 11, Z_MIN = -1, Z_MAX = 6;
const std::string SAVE_PATH = "/home/lj/文档/DebugfilterworldBACK/";  // 保存路径

struct Point {
    float x, y, z;
};

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
                    if (data[ix][iy][iz].size() > 10) {
                        result.emplace_back(ix, iy, iz, data[ix][iy][iz]);
                    }
                }
            }
        }
        return result;
    }
};

// 读取 CSV 文件
void processCSV(const std::string &filename, std::vector<Point> &cloud) {
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        float x, y, z;
        char comma;
        ss >> x >> comma >> y >> comma >> z;
        // std::cout << "xyz=" << x << "  " << y << "  " << z << std::endl;
        cloud.push_back({x, y, z});
    }
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

// 将点云数据保存到 CSV 文件
void saveToCSV(const std::unordered_map<int, std::vector<Point>> &classifiedClouds) {
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

void savePointCloudsToCSV(const std::vector<std::vector<Point>>& clouds, const std::string& filename) {
    // 创建一个文件流对象，打开文件进行写入
    std::ofstream file(filename);
    
    // 检查文件是否成功打开
    if (!file.is_open()) {
        std::cerr << "无法打开文件!" << std::endl;
        return;
    }
    
    // 写入数据
    for (size_t i = 0; i < clouds.size(); ++i) {
        for (size_t j = 0; j < clouds[i].size(); ++j) {
            const Point& p = clouds[i][j];
            file << p.x << "," << p.y << "," << p.z << std::endl;
        }
        // 每个点云之间插入空行（可选）
        file << std::endl;
    }
    
    // 关闭文件
    file.close();
    
    std::cout << "数据已保存到 " << filename << std::endl;
}

int main() {
    Grid grid;
    std::vector<std::vector<Point>> clouds;  // 每个文件的点云

    // 指定包含 CSV 文件的文件夹路径
    std::string folder_path = "/home/lj/文档/Debugfilterworld";
    // 获取文件夹中的所有 CSV 文件
    std::vector<std::string> files = getCSVFilesInDirectory(folder_path);

    
    // 读取所有文件
    for (int i = 0; i < files.size(); ++i) {
        std::vector<Point> cloud;
        processCSV(files[i], cloud);
        clouds.push_back(cloud);
        // if(i==0){
        //     savePointCloudsToCSV(clouds, "/home/lj/文档/pointclouds0.csv");
        // }
        for (const Point &point : cloud) {
            grid.fill(point.x, point.y, point.z, i);  // 每个文件的索引值从0开始
        }
    }

    
    // 获取多值网格
    auto multiValueGrids = grid.getGridsWithMultipleValues();

    
    // 分类点
    std::unordered_map<int, std::vector<Point>> classifiedClouds;
    for (int i = 0; i < files.size(); ++i) {
        classifyPoints(clouds[i], {i}, multiValueGrids, grid, classifiedClouds);
    }

    // 保存分类后的点云到文件
    // const std::string savePath = "/home/lj/文档/Debugfilterworld"; // 保存路径
    // 保存分类后的点云到 CSV 文件
    saveToCSV(classifiedClouds);
    
    // for (const auto &[frame, points] : classifiedClouds) {
    //     savePointCloudToCSV(savePath, frame, points);
    // }

    
    // 输出分类点云
    for (const auto &[frame, points] : classifiedClouds) {
        std::cout << "Frame " << frame << ":\n";
        for (const Point &point : points) {
            std::cout << point.x << ", " << point.y << ", " << point.z << "\n";
        }
    }
    

    return 0;
}
*/