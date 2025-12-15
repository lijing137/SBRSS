// map_ssim.cpp
#include <bits/stdc++.h>
#include <algorithm>

using Grid = std::vector<std::vector<double>>;

// 反射边界访问（symmetric padding）
static inline int reflectIndex(int x, int n) {
    if (n == 1) return 0;
    while (x < 0 || x >= n) {
        if (x < 0) x = -x - 1;
        if (x >= n) x = 2*n - x - 1;
    }
    return x;
}

template <typename T>
T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : (v > hi) ? hi : v;
}


// ---- 带权重的局部统计：权重 w = R + G（连续，无阈值）----
struct LocalStatsW { double muR, muG, varR, varG, covRG; };

LocalStatsW localStatsWeighted(const Grid& R, const Grid& G, int i, int j, int halfw) {
    const int H = (int)R.size(), W = (int)R[0].size();
    double sumW = 0, sumWR = 0, sumWG = 0, sumWRR = 0, sumWGG = 0, sumWRG = 0;

    for (int di = -halfw; di <= halfw; ++di) {
        int r = std::max(0, std::min(H - 1, i + di));      // 用零填充更干净；避免反射扩散
        for (int dj = -halfw; dj <= halfw; ++dj) {
            int c = std::max(0, std::min(W - 1, j + dj));
            double a = R[r][c];
            double b = G[r][c];
            double w = a + b;                    // 连续权重：候选/稳定处权重大，零零格子权重=0
            sumW   += w;
            sumWR  += w * a;
            sumWG  += w * b;
            sumWRR += w * a * a;
            sumWGG += w * b * b;
            sumWRG += w * a * b;
        }
    }

    // 若窗口内全是零零格子，回退到无权或返回“缺测”
    if (sumW <= 1e-12) {
        return {0,0,0,0,0}; // 调用方应识别这种情况
    }

    double muR = sumWR / sumW;
    double muG = sumWG / sumW;
    double varR = sumWRR / sumW - muR * muR;
    double varG = sumWGG / sumW - muG * muG;
    double covRG = sumWRG / sumW - muR * muG;
    return {muR, muG, varR, varG, covRG};
}

Grid ssimMapWeighted(const Grid& R, const Grid& G, int window = 3) {
    if (R.empty() || R[0].empty()) throw std::runtime_error("Empty grid.");
    const int H = (int)R.size(), W = (int)R[0].size();
    if ((int)G.size() != H || (int)G[0].size() != W) throw std::runtime_error("size mismatch");
    if (window % 2 == 0) ++window;
    int halfw = window / 2;

    // 概率图动态范围 L≈1，标准 SSIM 稳定项
    const double C1 = 1e-4;   // (0.01)^2
    const double C2 = 9e-4;   // (0.03)^2

    Grid S(H, std::vector<double>(W, 0.0));
    // for (int i = 0; i < H; ++i) {
    //     for (int j = 0; j < W; ++j) {
    //         auto st = localStatsWeighted(R, G, i, j, halfw);
    //         // 若窗口内 sumW≈0，说明没有候选/参考支持，令 SSIM=0（中性）
    //         if (st.varR==0 && st.varG==0 && st.muR==0 && st.muG==0) { S[i][j] = 0.0; continue; }

    //         double num_l  = 2.0 * st.muR * st.muG + C1;
    //         double den_l  = (st.muR*st.muR + st.muG*st.muG + C1);
    //         double num_cs = 2.0 * st.covRG + C2;
    //         double den_cs = (st.varR + st.varG + C2);
    //         double ssim = (den_l > 0 && den_cs > 0) ? (num_l/den_l) * (num_cs/den_cs) : 0.0;

    //         // 限幅
    //         if (ssim > 1.0) ssim = 1.0;
    //         if (ssim < -1.0) ssim = -1.0;
    //         S[i][j] = ssim;
    //     }
    // }
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            if (R[i][j] <= 0.1 || G[i][j] <= 0.1) {  // 中心像素不共现 -> 不评估
                S[i][j] = 0.0;
                continue;
            }
            auto st = localStatsWeighted(R, G, i, j, halfw);
            if (st.varR==0 && st.varG==0 && st.muR==0 && st.muG==0) { S[i][j] = 0.0; continue; }

            double num_l  = 2.0 * st.muR * st.muG + C1;
            double den_l  = st.muR*st.muR + st.muG*st.muG + C1;
            double num_cs = 2.0 * st.covRG + C2;
            double den_cs = st.varR + st.varG + C2;
            S[i][j] = (den_l > 0 && den_cs > 0) ? (num_l/den_l) * (num_cs/den_cs) : 0.0;
            // S[i][j] = std::clamp(S[i][j], -1.0, 1.0);
            S[i][j] = clamp(S[i][j], -1.0, 1.0);

        }
    }

    return S;
}


// 逐像素 MAP：返回每个栅格“正常”后验概率
Grid posteriorNormalFromSSIM(const Grid& S, double lambda = 5.0, double prior = 0.5) {
    const int H = (int)S.size(), W = (int)S[0].size();
    if (!(prior > 0.0 && prior < 1.0)) throw std::runtime_error("prior must be in (0,1).");
    Grid P(H, std::vector<double>(W, 0.0));
    const double logpi = std::log(prior);
    const double log1m = std::log(1.0 - prior);

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            double s = S[i][j];
            // ln post up to constant:
            double ln_norm = lambda * s + logpi;
            double ln_refl = lambda * (1.0 - s) + log1m;
            // p_norm = 1 / (1 + exp(ln_refl - ln_norm))
            double diff = ln_refl - ln_norm;
            // 数值稳定的 sigmoid
            double p_norm;
            if (diff > 50)       p_norm = std::exp(-diff);           // ~0
            else if (diff < -50) p_norm = 1.0 - std::exp(diff);      // ~1
            else                 p_norm = 1.0 / (1.0 + std::exp(diff));
            P[i][j] = p_norm;
        }
    }
    return P;
}

// 利用后验做 MAP 决策：true=normal, false=reflective
std::vector<std::vector<bool>> mapDecision(const Grid& Pnorm) {
    const int H = (int)Pnorm.size(), W = (int)Pnorm[0].size();
    std::vector<std::vector<bool>> M(H, std::vector<bool>(W, false));
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            M[i][j] = (Pnorm[i][j] >= 0.5); // 不是阈值，而是 MAP 的后验比较（等价于 argmax）
    return M;
}

// 小工具：打印
template<class T>
void printGrid(const std::string& name, const std::vector<std::vector<T>>& A, int precision=3) {
    std::cout << name << " (H=" << A.size() << ", W=" << (A.empty()?0:A[0].size()) << ")\n";
    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(precision);
    for (const auto& r : A) {
        std::cout << "[ ";
        for (auto v : r) std::cout << v << " ";
        std::cout << "]\n";
    }
    std::cout << std::defaultfloat;
}

int main() {
    // ======= 示例数据 =======
    // R: 历史稳定占据（接近真实静态表面），G: 被“去除的点”形成的候选占据
    // 实际使用时，把你真实的 R_t 和 G_t 填进来，取值在 [0,1]
    const int H = 12, W = 16;
    Grid R(H, std::vector<double>(W, 0.1));
    Grid G(H, std::vector<double>(W, 0.0));

    // 造一个“水平墙面”参考：中间一带为高占据概率（真实静态）
    for (int i = 5; i <= 6; ++i)
        for (int j = 2; j < W-11; ++j)
            R[i][j] = 0.4;

    for (int i = 5; i <= 6; ++i)
        for (int j = 9; j < W-2; ++j)
            R[i][j] = 0.4;        

    // G：其中一部分与 R 对齐（被误删的正常点回来了），另一部分是“游走的反射点”（稀疏、漂移）
    for (int i = 5; i <= 6; ++i)
        for (int j = 3; j < W-3; ++j)
            G[i][j] = 0.4;   // 这块应被判为“正常”

    // 随机加一些“反射游走”的小点
    // std::mt19937 rng(42);
    // std::uniform_int_distribution<int> uiH(0, H-1), uiW(0, W-1);
    // for (int k = 0; k < 20; ++k) G[uiH(rng)][uiW(rng)] = 0.2;

    // ======= 参数 =======
    const int win = 3;         // SSIM 窗口（奇数）
    const double lambda = 5.0; // 似然尺度（不是阈值）
    const double prior  = 0.5; // 正常先验（可用历史稳定估计）

    // ======= 计算 =======
    auto S  = ssimMapWeighted(R, G, win);                    // SSIM ∈ [-1,1]
    auto Pn = posteriorNormalFromSSIM(S, lambda, prior); // 每格“正常”的后验概率
    auto M  = mapDecision(Pn);                       // MAP 决策（normal / reflective）

    // ======= 输出 =======
    printGrid("R", R, 3);
    printGrid("G", G, 3);
    printGrid("SSIM", S, 3);
    printGrid("Posterior(normal)", Pn, 3);

    std::cout << "MAP decision (1=normal, 0=reflective)\n";
    for (int i = 0; i < H; ++i) {
        std::cout << "[ ";
        for (int j = 0; j < W; ++j) std::cout << (M[i][j] ? 1 : 0) << " ";
        std::cout << "]\n";
    }

    return 0;
}
