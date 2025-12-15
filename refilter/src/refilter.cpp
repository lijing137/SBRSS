#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <filesystem>
#include <pcl/filters/voxel_grid.h>
#include <chrono>

namespace fs = std::filesystem;

struct Pt4 { double x,y,z; double intensity; };
using Cloud = std::vector<Pt4>;
using Grid3D = std::vector<std::vector<std::vector<double>>>; // [Z][Y][X]
using Mask3D = std::vector<std::vector<std::vector<bool>>>;

namespace Params {
    // 输入文件夹（末尾不带斜杠）
    const std::string kFolderReflect = "/home/li/denoise_ws/src/refilter/Debugnoise";
    const std::string kFolderStatic = "/home/li/denoise_ws/src/refilter/scans.pcd";
    // 输出文件
    const std::string kOutCSV = "/home/li/denoise_ws/src/refilter/out_filtered.csv";
    // 体素大小（米）
    const double kVoxel = 0.4;     // change 0.4
    // 网格体素数量上限（避免意外超大内存）
    const size_t kMaxTotalVoxels = 2600ull*2600ull*2600ull; // ~27M
    // const size_t kMaxTotalVoxels = 400ull*400ull*400ull; // ~27M
    // SSIM 立方窗口（奇数）
    const int kWindow = 5;
    // SSIM 稳定常数（概率域 L≈1）
    const double kC1 = 1e-4;        // (0.01)^2
    const double kC2 = 9e-4;        // (0.03)^2

    // 共现门控（中心体素至少有一定占据）
    const double kCoOccurThresh = 0.50;

    // 后验参数
    const double kLambda = 5.0;
    const double kPrior  = 0.5;     // (0,1)

    const std::string kOutCSVSelected = "/home/lj/temp_ws/src/refilter/selected_points.csv";
    const std::string kOutCSVRemain = "/home/lj/temp_ws/src/refilter/remain_points.csv";

}

static bool hasSuffix(const std::string& s, const std::string& suf) {
    if (s.size() < suf.size()) return false;
    auto tolow = [](char c){ return (char)std::tolower((unsigned char)c); };
    for (size_t i=0;i<suf.size();++i) {
        if (tolow(s[s.size()-suf.size()+i]) != tolow(suf[i])) return false;
    }
    return true;
}

static bool isFile(const std::string& path) {
    struct stat sb{}; 
    return (stat(path.c_str(), &sb)==0) && S_ISREG(sb.st_mode);
}

static std::vector<std::string> listCSV(const std::string& folder) {
    std::vector<std::string> files;
    DIR* dir = opendir(folder.c_str());
    if (!dir) return files;
    dirent* ent;
    while ((ent=readdir(dir))!=nullptr) {
        std::string name = ent->d_name;
        if (name=="." || name=="..") continue;
        if (hasSuffix(name, ".csv")) {
            std::string full = folder + "/" + name;
            if (isFile(full)) files.push_back(full);
        }
    }
    closedir(dir);
    std::sort(files.begin(), files.end()); // 按字典序，适配 pc_noise_0.csv, pc_noise_1.csv...
    return files;
}

static bool parseCSVRowXYZI(const std::string& line, Pt4& p) {
    // 期望 4 列：x,y,z,intensity
    // 允许空白；不处理引号
    std::stringstream ss(line);
    std::string token;
    std::vector<std::string> cols;
    while (std::getline(ss, token, ',')) {
        // 去空白
        size_t b=0,e=token.size();
        while (b<e && std::isspace((unsigned char)token[b])) ++b;
        while (e>b && std::isspace((unsigned char)token[e-1])) --e;
        cols.emplace_back(token.substr(b,e-b));
    }
    if (cols.size()<3) return false;
    try {
        p.x = std::stod(cols[0]);
        p.y = std::stod(cols[1]);
        p.z = std::stod(cols[2]);
        p.intensity = (cols.size()>=4) ? std::stod(cols[3]) : 0.0;
    } catch (...) { return false; }
    return true;
}

static Cloud loadFolderCSV(const std::string& folder) {
    Cloud P; 
    auto files = listCSV(folder);
    size_t fileCount = files.size();
    std::cerr << "[IO] Folder: " << folder << ", csv files = " << fileCount << "\n";
    for (const auto& f : files) {
        std::ifstream fin(f);
        if (!fin) { std::cerr << "  - Skip (open fail): " << f << "\n"; continue; }
        std::string line;
        bool first=true;
        size_t good=0, bad=0;
        while (std::getline(fin, line)) {
            if (first) { // 跳过 header
                first=false; continue;
            }
            if (line.empty()) continue;
            Pt4 p{};
            if (parseCSVRowXYZI(line, p)) { P.push_back(p); ++good; }
            else ++bad;
        }
        // std::cerr << "  - " << f << "  ok=" << good << " bad=" << bad << "\n";
    }
    std::cerr << "[IO] Loaded points: " << P.size() << "\n";
    return P;
}

static Cloud fromPCL(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pclCloud){
    Cloud out; out.reserve(pclCloud->size());
    for (const auto& pt : pclCloud->points){
        Pt4 p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.intensity = pt.intensity;
        out.push_back(p);
    }
    return out;
}

// Cloud -> pcl::PointCloud<pcl::PointXYZI>
pcl::PointCloud<pcl::PointXYZI>::Ptr toPCL(const Cloud& c)
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    cloud->reserve(c.size());

    for (const auto& p : c) {
        pcl::PointXYZI pt;
        pt.x = static_cast<float>(p.x);         // 根据你的 Cloud 点类型修改
        pt.y = static_cast<float>(p.y);
        pt.z = static_cast<float>(p.z);
        pt.intensity = static_cast<float>(p.intensity);
        cloud->push_back(pt);
    }

    return cloud;
}

struct AABB {
    double minx=-10,miny=-10,minz=-5;
    double maxx=+10,maxy=+10,maxz=+5;
    void expand(const Pt4& p){
        minx=std::min(minx,p.x); miny=std::min(miny,p.y); minz=std::min(minz,p.z);
        maxx=std::max(maxx,p.x); maxy=std::max(maxy,p.y); maxz=std::max(maxz,p.z);
    }
    bool valid() const { return minx<=maxx && miny<=maxy && minz<=maxz; }
};

struct VoxelSpec {
    double vx,vy,vz;
    size_t nx,ny,nz;
    double ox,oy,oz;

};

static size_t ceilDiv(double span, double v) {
    if (span<=0) return 1;
    return (size_t)std::max(1.0, std::ceil(span / std::max(v,1e-9)));
}

static VoxelSpec makeSpec(const AABB& bb, double voxel) {
    VoxelSpec s;
    s.vx=s.vy=s.vz = std::max(1e-9, voxel);
    s.ox=bb.minx; s.oy=bb.miny; s.oz=bb.minz;
    s.nx=ceilDiv(bb.maxx-bb.minx, s.vx);
    s.ny=ceilDiv(bb.maxy-bb.miny, s.vy);
    s.nz=ceilDiv(bb.maxz-bb.minz, s.vz);
    return s;
}

inline size_t idx(int x, int y, int z, const VoxelSpec& spec) {
    return (size_t)z * spec.nx * spec.ny + (size_t)y * spec.nx + (size_t)x;
}

static inline bool pointToIndex(const VoxelSpec& s, const Pt4& p, int& ix,int& iy,int& iz){
    ix = (int)std::floor((p.x - s.ox)/s.vx);
    iy = (int)std::floor((p.y - s.oy)/s.vy);
    iz = (int)std::floor((p.z - s.oz)/s.vz);
    return (ix>=0 && ix<(int)s.nx && iy>=0 && iy<(int)s.ny && iz>=0 && iz<(int)s.nz);
}

static Grid3D voxelizeDensity(const Cloud& P, const VoxelSpec& s) {
    Grid3D grid(s.nz, std::vector<std::vector<double>>(s.ny, std::vector<double>(s.nx, 0.0)));
    std::vector<int> counts(s.nx*s.ny*s.nz, 0);
    auto lid = [&](int x,int y,int z)->size_t { return (size_t)z*s.ny*s.nx + (size_t)y*s.nx + (size_t)x; };

    int ix,iy,iz; int maxc=0;
    for (const auto& pt : P) {
        if (pointToIndex(s, pt, ix,iy,iz)) {
            int& c = counts[lid(ix,iy,iz)];
            c += 1;
            if (c>maxc) maxc=c;
        }
    }
    // std::cout << "maxc = " << maxc << std::endl;

    if (maxc == 0)
        return grid;

    for (size_t z=0; z<s.nz; ++z)
        for (size_t y=0; y<s.ny; ++y)
            for (size_t x=0; x<s.nx; ++x) {
                int c = counts[lid((int)x,(int)y,(int)z)];
                grid[z][y][x] = (double)c / (double)maxc;
            }
    return grid;
}


Grid3D computeIntensitySSIM_Voxelwise(
    const Cloud& cloudA,
    const Cloud& cloudB,
    const VoxelSpec& spec)
{
    // 预计算每个体素包含的点索引列表
    std::vector<std::vector<int>> voxelA(spec.nz * spec.ny * spec.nx);
    std::vector<std::vector<int>> voxelB(spec.nz * spec.ny * spec.nx);

    // 点放入对应体素
    auto voxelIndex = [&](const Pt4& p) -> int64_t {
        int x = std::floor((p.x - spec.ox) / spec.vx);
        int y = std::floor((p.y - spec.oy) / spec.vy);
        int z = std::floor((p.z - spec.oz) / spec.vz);
        if (x < 0 || x >= spec.nx || y < 0 || y >= spec.ny || z < 0 || z >= spec.nz) return -1;
        return static_cast<int64_t>(z) * spec.nx * spec.ny + y * spec.nx + x;
    };

    // 体素填充
    for (int i = 0; i < cloudA.size(); ++i) {
        int vid = voxelIndex(cloudA[i]);
        if (vid >= 0) voxelA[vid].push_back(i);
    }
    for (int i = 0; i < cloudB.size(); ++i) {
        int vid = voxelIndex(cloudB[i]);
        if (vid >= 0) voxelB[vid].push_back(i);
    }

    // SSIM 计算
    const double C1 = 1e-4;
    const double C2 = 9e-4;

    // 存储每体素 SSIM 结果
    Grid3D ssim_int(spec.nz,
    std::vector<std::vector<double>>(spec.ny,
        std::vector<double>(spec.nx, 0.0)));
    
    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {

                int vid = z * spec.nx * spec.ny + y * spec.nx + x;
                auto& idxA = voxelA[vid];
                auto& idxB = voxelB[vid];

                if (idxA.empty() || idxB.empty()) {
                    ssim_int[z][y][x] = 0.0;      // 该体素无共同点
                    continue;
                }

                // 收集该体素内的强度
                std::vector<double> fA, fB;
                for (int ia : idxA) fA.push_back(cloudA[ia].intensity);
                for (int ib : idxB) fB.push_back(cloudB[ib].intensity);

                // 点数不一致时，建议采用 min 匹配（也可插值）
                int N = std::min(fA.size(), fB.size());
                if (N < 2) {
                    ssim_int[z][y][x] = 0.0;
                    continue;
                }

                fA.resize(N);
                fB.resize(N);

                // 均值
                double meanA = std::accumulate(fA.begin(), fA.end(), 0.0) / N;
                double meanB = std::accumulate(fB.begin(), fB.end(), 0.0) / N;

                // 方差及协方差
                double varA = 0.0, varB = 0.0, covAB = 0.0;
                for (int i = 0; i < N; ++i) {
                    varA += (fA[i] - meanA) * (fA[i] - meanA);
                    varB += (fB[i] - meanB) * (fB[i] - meanB);
                    covAB += (fA[i] - meanA) * (fB[i] - meanB);
                }
                varA /= (N - 1);
                varB /= (N - 1);
                covAB /= (N - 1);

                // SSIM
                double numerator = (2 * meanA * meanB + C1) * (2 * covAB + C2);
                double denominator = (meanA * meanA + meanB * meanB + C1) * (varA + varB + C2);

                ssim_int[z][y][x] = numerator / denominator;
            }
        }
    }
    return ssim_int;
}

double computeSSIM1D(double meanA, double meanB,
                     double varA,double varB,  
                     double covAB, double C1, double C2){
    double numerator = (2 * meanA * meanB + C1) * (2 * covAB + C2);
    double denominator = (meanA * meanA + meanB * meanB + C1) * (varA + varB + C2);
    double ssim_cor = numerator / denominator;
    return ssim_cor;
}

Grid3D computeCoordSSIM_Voxelwise(
    const Cloud& cloudA,
    const Cloud& cloudB,
    const VoxelSpec& spec)
{
    // 预计算每个体素包含的点索引列表
    std::vector<std::vector<int>> voxelA(spec.nz * spec.ny * spec.nx);
    std::vector<std::vector<int>> voxelB(spec.nz * spec.ny * spec.nx);

    // 点放入对应体素
    auto voxelIndex = [&](const Pt4& p) -> int64_t {
        int x = std::floor((p.x - spec.ox) / spec.vx);
        int y = std::floor((p.y - spec.oy) / spec.vy);
        int z = std::floor((p.z - spec.oz) / spec.vz);
        if (x < 0 || x >= spec.nx || y < 0 || y >= spec.ny || z < 0 || z >= spec.nz) return -1;
        return static_cast<int64_t>(z) * spec.nx * spec.ny + y * spec.nx + x;
    };

    // 体素填充
    for (int i = 0; i < cloudA.size(); ++i) {
        int vid = voxelIndex(cloudA[i]);
        if (vid >= 0) voxelA[vid].push_back(i);
    }
    for (int i = 0; i < cloudB.size(); ++i) {
        int vid = voxelIndex(cloudB[i]);
        if (vid >= 0) voxelB[vid].push_back(i);
    }

    // SSIM 计算
    const double C1 = 1e-4;
    const double C2 = 9e-4;

    // 存储每体素 SSIM 结果
    Grid3D ssim_cor(spec.nz,
    std::vector<std::vector<double>>(spec.ny,
        std::vector<double>(spec.nx, 0.0)));

    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {

                int vid = z * spec.nx * spec.ny + y * spec.nx + x;
                auto& idxA = voxelA[vid];
                auto& idxB = voxelB[vid];

                if (idxA.empty() || idxB.empty()) {
                    ssim_cor[z][y][x] = 0.0;      // 该体素无共同点
                    continue;
                }

                // 收集该体素内的点坐标
                std::vector<double> xA, yA, zA;
                std::vector<double> xB, yB, zB;

                for (int ia : idxA) {
                    xA.push_back(cloudA[ia].x);
                    yA.push_back(cloudA[ia].y);
                    zA.push_back(cloudA[ia].z);
                }

                for (int ib : idxB) {
                    xB.push_back(cloudB[ib].x);
                    yB.push_back(cloudB[ib].y);
                    zB.push_back(cloudB[ib].z);
                }

                // 点数不一致时，建议采用 min 匹配（也可插值）
                int N = std::min(xA.size(), xB.size());
                if (N < 2) {
                    ssim_cor[z][y][x] = 0.0;
                    continue;
                }

                // resize 坐标向量
                xA.resize(N);
                yA.resize(N);
                zA.resize(N);
                xB.resize(N);
                yB.resize(N);
                zB.resize(N);

                // 均值
                double meanxA = std::accumulate(xA.begin(), xA.end(), 0.0) / N;
                double meanyA = std::accumulate(yA.begin(), yA.end(), 0.0) / N;
                double meanzA = std::accumulate(zA.begin(), zA.end(), 0.0) / N;
                
                double meanxB = std::accumulate(xB.begin(), xB.end(), 0.0) / N;
                double meanyB = std::accumulate(yB.begin(), yB.end(), 0.0) / N;
                double meanzB = std::accumulate(zB.begin(), zB.end(), 0.0) / N;

                // 方差及协方差
                double varAx = 0.0, varBx = 0.0, covABx = 0.0;
                double varAy = 0.0, varBy = 0.0, covABy = 0.0;
                double varAz = 0.0, varBz = 0.0, covABz = 0.0;

                for (int i = 0; i < N; ++i) {
                    varAx += (xA[i] - meanxA) * (xA[i] - meanxA);
                    varAy += (yA[i] - meanyA) * (yA[i] - meanyA);
                    varAz += (zA[i] - meanzA) * (zA[i] - meanzA);

                    varBx += (xB[i] - meanxB) * (xB[i] - meanxB);
                    varBy += (yB[i] - meanyB) * (yB[i] - meanyB);
                    varBz += (zB[i] - meanzB) * (zB[i] - meanzB);

                    covABx += (xA[i] - meanxA) * (xB[i] - meanxB);
                    covABy += (yA[i] - meanyA) * (yB[i] - meanyB);
                    covABz += (zA[i] - meanzA) * (zB[i] - meanzB);

                }
                
                varAx /= (N - 1);
                varAy /= (N - 1);
                varAz /= (N - 1);

                varBx /= (N - 1);
                varBy /= (N - 1);
                varBz /= (N - 1);

                covABx /= (N - 1);
                covABy /= (N - 1);
                covABz /= (N - 1);

                double ssim_cor_x = computeSSIM1D(meanxA, meanxB, varAx, varBx, covABx, C1, C2);
                double ssim_cor_y = computeSSIM1D(meanyA, meanyB, varAy, varBy, covABy, C1, C2);
                double ssim_cor_z = computeSSIM1D(meanzA, meanzB, varAz, varBz, covABz, C1, C2);

                ssim_cor[z][y][x] = (ssim_cor_x + ssim_cor_y + ssim_cor_z) / 3.0;
            }
        }
    }

    return ssim_cor;
}

int main(){

    auto t0 = std::chrono::steady_clock::now();
    
    // 1) 读取两个文件夹中的 CSV 点云
    Cloud Gcloud = loadFolderCSV(Params::kFolderReflect);

    // 创建一个点云对象，用于存储输入点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // 读取点云文件
    if (pcl::io::loadPCDFile<pcl::PointXYZI>(Params::kFolderStatic, *cloud) == -1) {
        std::cerr << "Couldn't read the file: " << Params::kFolderStatic << std::endl;
        return -1;
    }
    Cloud Rcloud = fromPCL(cloud);

    const float voxel_leaf = 0.01f;  // 比如 0.1 m，自行调整
    pcl::PointCloud<pcl::PointXYZI>::Ptr gcloud_pcl = toPCL(Gcloud);
    pcl::PointCloud<pcl::PointXYZI>::Ptr gcloud_ds(new pcl::PointCloud<pcl::PointXYZI>);

    {
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setInputCloud(gcloud_pcl);
        vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
        vg.filter(*gcloud_ds);
    }
    // 重新赋值为降采样后的 Cloud
    Cloud Gcloud_ds = fromPCL(gcloud_ds);

    // downsample rcloud
    pcl::PointCloud<pcl::PointXYZI>::Ptr rcloud_ds(new pcl::PointCloud<pcl::PointXYZI>);

    {
        pcl::VoxelGrid<pcl::PointXYZI> vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(voxel_leaf, voxel_leaf, voxel_leaf);
        vg.filter(*rcloud_ds);
    }
    // 重新赋值为降采样后的 Cloud
    Cloud Rcloud_ds = fromPCL(rcloud_ds);

    Gcloud = Gcloud_ds;
    Rcloud = Rcloud_ds;

    // 2) 计算联合包围盒
    AABB bb;
    for (auto& p:Gcloud) bb.expand(p);
    for (auto& p:Rcloud) bb.expand(p);
    if (!bb.valid()){ std::cerr<<"[Error] 无效包围盒。\n"; return 2; }

    // 3) 构建体素网格规格
    VoxelSpec spec = makeSpec(bb, Params::kVoxel);
    const size_t total = spec.nx*spec.ny*spec.nz;
    std::cerr<<"[Grid] dims nx,ny,nz = ("<<spec.nx<<","<<spec.ny<<","<<spec.nz<<")"
                <<", total="<< total <<", voxel="<< Params::kVoxel << " m\n";
    if (total > Params::kMaxTotalVoxels){
        std::cerr<<"[Error] 体素数量过大（"<<total<<" > "<<Params::kMaxTotalVoxels<<"）。请增大体素尺寸。\n";
        return 3;
    }

    // 5) 统计每个体素被多少帧命中 -> Frame Count Grid3D FC
    // 初始化：
    Grid3D FC(
        spec.nz,  // Z 方向大小
        std::vector<std::vector<double>>(
            spec.ny,  // Y 方向大小
            std::vector<double>(spec.nx, 0.0)  // X 方向大小
        )
    );// 和 G/R 同规格的网格，构造方式按你自己的 Grid3D 定义来

    for (int iz = 0; iz < spec.nz; ++iz)
        for (int iy = 0; iy < spec.ny; ++iy)
            for (int ix = 0; ix < spec.nx; ++ix)
                FC[iz][iy][ix] = 0.0;

    // 遍历 Params::kFolderReflect 下所有 pc_noise_***.csv
    for (const auto& entry : fs::directory_iterator(Params::kFolderReflect)) {
        if (!entry.is_regular_file()) continue;
        const auto& path = entry.path();

        // 只处理 .csv 且文件名形如 pc_noise_*.csv（可按需要调整过滤条件）
        if (path.extension() != ".csv") continue;
        const std::string fname = path.filename().string();
        if (fname.rfind("pc_noise_", 0) != 0) continue;   // 不是以 "pc_noise_" 开头就跳过

        Cloud P; 
        std::ifstream fin(path.string());
        if (!fin) { std::cerr << "  - Skip (open fail): " << path.string() << "\n"; continue; }
        std::string line;
        bool first=true;
        size_t good=0, bad=0;
        while (std::getline(fin, line)) {
            if (first) { // 跳过 header
                first=false; continue;
            }
            if (line.empty()) continue;
            Pt4 p{};
            if (parseCSVRowXYZI(line, p)) { P.push_back(p); ++good; }
            else ++bad;
        }

        // 单帧点云体素化 —— 仍然使用上面求得的 spec（统一网格）
        Grid3D occ = voxelizeDensity(P, spec);

        // 对该帧：凡是该体素有点（密度 > 0），就在 FC 中 +1
        for (int iz = 0; iz < spec.nz; ++iz) {
            for (int iy = 0; iy < spec.ny; ++iy) {
                for (int ix = 0; ix < spec.nx; ++ix) {
                    if (occ[iz][iy][ix] > 0.0f) {
                        FC[iz][iy][ix] += 1.0f;
                        
                    }
                }
            }
        }
        
    }

    double max_val = -std::numeric_limits<double>::infinity();
    for (const auto& plane : FC) {
        for (const auto& row : plane) {
            for (double v : row) {
                if (v > max_val) {
                    max_val = v;
                }
            }
        }
    }

    for (int iz = 0; iz < spec.nz; ++iz) {
        for (int iy = 0; iy < spec.ny; ++iy) {
            for (int ix = 0; ix < spec.nx; ++ix) {
                // std::cout << "FC=" << FC[iz][iy][ix] << std::endl;
                FC[iz][iy][ix] /= max_val; // 转成概率
            }
        }
    }

    Grid3D ssim_int = computeIntensitySSIM_Voxelwise(Gcloud, Rcloud, spec);

    double max_val_int = -std::numeric_limits<double>::infinity();
    for (const auto& plane : ssim_int) {
        for (const auto& row : plane) {
            for (double v : row) {
                if (v > max_val_int) {
                    max_val_int = v;
                }
            }
        }
    }
    std::cout << "max_val_int 中的最大值 = " << max_val_int << std::endl;


    Grid3D ssim_cor = computeCoordSSIM_Voxelwise(Gcloud, Rcloud, spec);

    double max_val_cor = -std::numeric_limits<double>::infinity();
    for (const auto& plane : ssim_int) {
        for (const auto& row : plane) {
            for (double v : row) {
                if (v > max_val_cor) {
                    max_val_cor = v;
                }
            }
        }
    }
    std::cout << "max_val_cor 中的最大值 = " << max_val_cor << std::endl;

    Grid3D ssim(spec.nz,
            std::vector<std::vector<double>>(spec.ny,
                std::vector<double>(spec.nx, 0.0)));

    for (int z = 0; z < spec.nz; ++z) {
        for (int y = 0; y < spec.ny; ++y) {
            for (int x = 0; x < spec.nx; ++x) {
                // ssim[z][y][x] = 0.33 * FC[z][y][x] + 0.33 * ssim_int[z][y][x] + 0.30 * ssim_cor[z][y][x];
                // change
                // ssim[z][y][x] = FC[z][y][x];
                ssim[z][y][x] = ssim_cor[z][y][x];
                // ssim[z][y][x] = ssim_int[z][y][x];
                // ssim[z][y][x] = 0.5 * FC[z][y][x] + 0.5 * ssim_cor[z][y][x];
                // ssim[z][y][x] = std::max(FC[z][y][x], ssim_cor[z][y][x]);
            }
        }
    }

    std::vector<Pt4> selected_points;
    selected_points.reserve(Gcloud.size());

    std::vector<Pt4> remain_points;
    remain_points.reserve(Gcloud.size());

    for (const auto& p : Gcloud) {
        // 计算点所在体素索引
        int ix = static_cast<int>(std::floor((p.x - spec.ox) / spec.vx));
        int iy = static_cast<int>(std::floor((p.y - spec.oy) / spec.vy));
        int iz = static_cast<int>(std::floor((p.z - spec.oz) / spec.vz));

        // 越界直接跳过
        if (ix < 0 || iy < 0 || iz < 0 ||
            ix >= spec.nx || iy >= spec.ny || iz >= spec.nz) {
            continue;
        }

        // 查该体素的 ssim 值
        double v = ssim[iz][iy][ix];

        // 如果 ssim > 0.5，就把该点加入输出点云 change
        if (v > 0.6) { //0.5
            selected_points.push_back(p);
        }
        else{
            remain_points.push_back(p);
        }
    }    

    auto t1 = std::chrono::steady_clock::now();
    auto ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "over deletion 耗时: " << ms << " ms\n";
    
    // 把 selected_points 写入 CSV 文件
    std::ofstream ofs_select(Params::kOutCSVSelected);
    if (!ofs_select.is_open()) {
        std::cerr << "[Error] 无法打开输出文件: " << Params::kOutCSVSelected << std::endl;
    } else {
        // 写表头（可选）
        ofs_select << "x,y,z,intensity\n";

        for (const auto& p : selected_points) {
            ofs_select << p.x << ","
                << p.y << ","
                << p.z << ","
                << p.intensity << "\n";
        }
        ofs_select.close();
        std::cout << "保存筛选后的点云到: " << Params::kOutCSVSelected
                << "，共 " << selected_points.size() << " 个点\n";
    }

    // 把 remain_points 写入 CSV 文件
    std::ofstream ofs_remain(Params::kOutCSVRemain);
    if (!ofs_remain.is_open()) {
        std::cerr << "[Error] 无法打开输出文件: " << Params::kOutCSVRemain << std::endl;
    } else {
        // 写表头（可选）
        ofs_remain << "x,y,z,intensity\n";

        for (const auto& p : remain_points) {
            ofs_remain << p.x << ","
                << p.y << ","
                << p.z << ","
                << p.intensity << "\n";
        }
        ofs_remain.close();
        std::cout << "保存筛选后的点云到: " << Params::kOutCSVRemain
                << "，共 " << remain_points.size() << " 个点\n";
    }

    return 0;
}
