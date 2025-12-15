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
using Grid3D = std::vector<std::vector<std::vector<double>>>; // [Z][Y][X]
using Mask3D = std::vector<std::vector<std::vector<bool>>>;



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
    if (maxc==0) return grid;

    for (size_t z=0; z<s.nz; ++z)
        for (size_t y=0; y<s.ny; ++y)
            for (size_t x=0; x<s.nx; ++x) {
                int c = counts[lid((int)x,(int)y,(int)z)];
                grid[z][y][x] = (double)c / (double)maxc;
            }
    return grid;
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

template <typename T>
static inline T clampv(T v, T lo, T hi) { return (v<lo)?lo:((v>hi)?hi:v); }

struct LocalStatsW { double muR, muG, varR, varG, covRG; };

static LocalStatsW localStatsWeighted3D(const Grid3D& R, const Grid3D& G,
                                        int z0, int y0, int x0, int halfw)
{
    const int Z=(int)R.size(), Y=(int)R[0].size(), X=(int)R[0][0].size();
    double sumW=0, sumWR=0, sumWG=0, sumWRR=0, sumWGG=0, sumWRG=0;

    for (int dz=-halfw; dz<=halfw; ++dz){
        int z = clampv(z0+dz, 0, Z-1);
        for (int dy=-halfw; dy<=halfw; ++dy){
            int y = clampv(y0+dy, 0, Y-1);
            for (int dx=-halfw; dx<=halfw; ++dx){
                int x = clampv(x0+dx, 0, X-1);
                double a = R[z][y][x];
                double b = G[z][y][x];
                double w = a + b;
                sumW   += w;
                sumWR  += w * a;
                sumWG  += w * b;
                sumWRR += w * a * a;
                sumWGG += w * b * b;
                sumWRG += w * a * b;
            }
        }
    }
    if (sumW<=1e-12) return {0,0,0,0,0};
    double muR = sumWR / sumW;
    double muG = sumWG / sumW;
    double varR = sumWRR / sumW - muR*muR;
    double varG = sumWGG / sumW - muG*muG;
    double covRG = sumWRG / sumW - muR*muG;
    return {muR,muG,varR,varG,covRG};
}

static Grid3D ssimMapWeighted3D(const Grid3D& R, const Grid3D& G, int window){
    if (R.empty() || R[0].empty() || R[0][0].empty()) throw std::runtime_error("Empty grid.");
    const int Z=(int)R.size(), Y=(int)R[0].size(), X=(int)R[0][0].size();
    if ((int)G.size()!=Z || (int)G[0].size()!=Y || (int)G[0][0].size()!=X) throw std::runtime_error("size mismatch");
    if (window%2==0) ++window;
    int halfw = window/2;

    Grid3D S(Z, std::vector<std::vector<double>>(Y, std::vector<double>(X, 0.0)));

    for (int z=0; z<Z; ++z)
        for (int y=0; y<Y; ++y)
            for (int x=0; x<X; ++x){
                // 中心体素共现门控
                if (R[z][y][x] <= Params::kCoOccurThresh || G[z][y][x] <= Params::kCoOccurThresh) {
                    S[z][y][x] = 0.0;
                    continue;
                }
                auto st = localStatsWeighted3D(R,G,z,y,x,halfw);
                if (st.varR==0 && st.varG==0 && st.muR==0 && st.muG==0) { S[z][y][x]=0.0; continue; }

                double num_l  = 2.0*st.muR*st.muG + Params::kC1;
                double den_l  = st.muR*st.muR + st.muG*st.muG + Params::kC1;
                double num_cs = 2.0*st.covRG + Params::kC2;
                double den_cs = st.varR + st.varG + Params::kC2;

                double ssim = (den_l>0 && den_cs>0) ? (num_l/den_l)*(num_cs/den_cs) : 0.0;
                if (ssim>1.0) ssim=1.0;
                if (ssim<-1.0) ssim=-1.0;
                S[z][y][x]=ssim;
            }
    return S;
}

static Grid3D posteriorNormalFromSSIM3D(const Grid3D& S, double lambda, double prior){
    if (!(prior>0.0 && prior<1.0)) throw std::runtime_error("prior must be in (0,1).");
    const int Z=(int)S.size(), Y=(int)S[0].size(), X=(int)S[0][0].size();
    Grid3D P(Z, std::vector<std::vector<double>>(Y, std::vector<double>(X, 0.0)));

    const double logpi = std::log(prior), log1m = std::log(1.0-prior);
    for (int z=0; z<Z; ++z)
        for (int y=0; y<Y; ++y)
            for (int x=0; x<X; ++x){
                double s=S[z][y][x];
                double ln_norm = lambda*s + logpi;
                double ln_refl = lambda*(1.0 - s) + log1m;
                double diff = ln_refl - ln_norm;
                double p_norm;
                if (diff>50)       p_norm = std::exp(-diff);
                else if (diff<-50) p_norm = 1.0 - std::exp(diff);
                else               p_norm = 1.0/(1.0+std::exp(diff));
                P[z][y][x]=p_norm;
            }
    return P;
}

static Mask3D mapDecision3D(const Grid3D& Pnorm){
    const int Z=(int)Pnorm.size(), Y=(int)Pnorm[0].size(), X=(int)Pnorm[0][0].size();
    Mask3D M(Z, std::vector<std::vector<bool>>(Y, std::vector<bool>(X,false)));
    for (int z=0; z<Z; ++z)
        for (int y=0; y<Y; ++y)
            for (int x=0; x<X; ++x)
                M[z][y][x] = (Pnorm[z][y][x] >= 0.5);
    return M;
}

static Cloud filterByMask(const Cloud& G, const VoxelSpec& s, const Mask3D& M){
    Cloud out; out.reserve(G.size());
    int ix,iy,iz;
    for (const auto& p : G){
        if (pointToIndex(s, p, ix,iy,iz)) {
            if (M[(size_t)iz][(size_t)iy][(size_t)ix]) out.push_back(p);
        }
    }
    return out;
}

static bool saveCSV(const std::string& path, const Cloud& P){
    std::ofstream fout(path);
    if(!fout) return false;
    fout << "x,y,z,intensity\n";
    fout.setf(std::ios::fixed); fout<<std::setprecision(6);
    for (auto& p:P) {
        fout<<p.x<<","<<p.y<<","<<p.z<<","<<p.intensity<<"\n";
    }
    return true;
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
        Cloud Rcloud = fromPCL(cloud);

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

        // 4) 体素化 -> 概率体素 [0,1]
        Grid3D G = voxelizeDensity(Gcloud, spec);
        Grid3D R = voxelizeDensity(Rcloud, spec);

        // 5) 3D 加权 SSIM
        int window = Params::kWindow;
        if (window%2==0) { std::cerr<<"[Info] 窗口需为奇数，自动+1。\n"; ++window; }
        Grid3D S = ssimMapWeighted3D(R, G, window);

        // 6) 后验 / MAP
        Grid3D Pn = posteriorNormalFromSSIM3D(S, Params::kLambda, Params::kPrior);
        Mask3D M  = mapDecision3D(Pn);

        // 7) 用体素掩码筛选 G 的“正常”点
        Cloud G_keep = filterByMask(Gcloud, spec, M);
        std::cerr<<"[Keep] G 输入点数: "<<Gcloud.size()<<", 保留(判定为正常): "<<G_keep.size()<<"\n";

        // 8) 保存 CSV
        if (!saveCSV(Params::kOutCSV, G_keep)) {
            std::cerr<<"[Error] 保存失败: "<<Params::kOutCSV<<"\n"; return 4;
        }

        
        std::cerr<<"[OK] 已保存: "<<Params::kOutCSV<<"\n";
        return 0;
    } catch (const std::exception& e){
        std::cerr<<"[Exception] "<<e.what()<<"\n";
        return 10;
    }
}