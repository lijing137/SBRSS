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

// #include <pcl/point_cloud.h>
// #include <pcl/io/pcd_io.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/point_types.h>

struct Pt4 { double x,y,z; double intensity; };
using Cloud = std::vector<Pt4>;

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

namespace Params {
    // 输入文件夹（末尾不带斜杠）
    const std::string kFolderReflect = "/home/lj/temp_ws/src/refilter/Debugnoise";
    const std::string kFolderStatic = "/home/lj/temp_ws/src/refilter/scans.pcd";
    // 输出文件
    const std::string kOutCSV = "/home/lj/temp_ws/src/refilter/out_filtered.csv";
    // 体素大小（米）
    const double kVoxel = 0.05;     // 5 cm
    // 网格体素数量上限（避免意外超大内存）
    const size_t kMaxTotalVoxels = 300ull*300ull*300ull; // ~27M

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

struct AABB {
    double minx=+1e300,miny=+1e300,minz=+1e300;
    double maxx=-1e300,maxy=-1e300,maxz=-1e300;
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

static VoxelSpec makeSpec(const AABB& bb, double voxel) {
    VoxelSpec s;
    s.vx=s.vy=s.vz = std::max(1e-9, voxel);
    s.ox=bb.minx; s.oy=bb.miny; s.oz=bb.minz;
    s.nx=ceilDiv(bb.maxx-bb.minx, s.vx);
    s.ny=ceilDiv(bb.maxy-bb.miny, s.vy);
    s.nz=ceilDiv(bb.maxz-bb.minz, s.vz);
    return s;
}

int main(){
    try{
        // 1) 读取两个文件夹中的 CSV 点云
        Cloud Gcloud = loadFolderCSV(Params::kFolderReflect);

        // 创建一个点云对象，用于存储输入点云
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        // 读取点云文件
        if (pcl::io::loadPCDFile<pcl::PointXYZI>(Params::kFolderStatic, *cloud) == -1) {
            std::cerr << "Couldn't read the file: " << Params::kFolderStatic << std::endl;
            return -1;
        }

        // 2) 计算联合包围盒
        AABB bb;
        for (auto& p:Gcloud) bb.expand(p);
        if (!bb.valid()){ std::cerr<<"[Error] 无效包围盒。\n"; return 2; }

        // 3) 构建体素网格规格
        VoxelSpec spec = makeSpec(bb, Params::kVoxel);
        const size_t total = spec.nx*spec.ny*spec.nz;
        std::cerr<<"[Grid] dims nx,ny,nz = ("<<spec.nx<<","<<spec.ny<<","<<spec.nz<<")"
                 <<", total="<< total <<", voxel="<< Params::kVoxel << " m\n";

        
        std::cerr<<"[OK] 已保存: "<<Params::kOutCSV<<"\n";
        return 0;
    } catch (const std::exception& e){
        std::cerr<<"[Exception] "<<e.what()<<"\n";
        return 10;
    }
}
