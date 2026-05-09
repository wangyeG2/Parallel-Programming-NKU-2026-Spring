// pq_simd.h
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <utility>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <omp.h>
#include <arm_neon.h>
#include "simd8float32.h"   // 复用已有的 8‑float 封装

struct PQIndex {
    size_t M;           // 子空间数 = 4
    size_t Ks;          // 每个子空间的聚类中心数 = 256
    size_t subdim;      // 每个子空间的维度 = 24
    size_t total_dim;   // 原始维度 = 96

    // 码本：M * Ks * subdim 个 float，按 [m][c][*] 存储
    std::vector<float> codebook;

    // 所有 base 向量的 PQ 编码：base_number * M 个 uint8_t，按 [i][m] 存储
    std::vector<uint8_t> codes;

    PQIndex() = default;
};

// 朴素 KMeans 的 SIMD 内积距离计算辅助函数
inline float simd_dot_product(const float* x, const float* y, size_t dim) {
    assert(dim % 8 == 0);
    simd8float32 sum(0.0f);
    for (size_t i = 0; i < dim; i += 8) {
        simd8float32 sx(x + i), sy(y + i);
        sum += sx * sy;
    }
    return sum.sum();
}

// 构建 PQ 索引
void build_pq_index(const float* base, size_t base_number, size_t vecdim,
                    PQIndex& index, size_t M = 4, size_t Ks = 256) {
    index.M = M;
    index.Ks = Ks;
    index.subdim = vecdim / M;
    index.total_dim = vecdim;
    assert(vecdim % M == 0 && index.subdim % 8 == 0);  // 保证 8 对齐

    // 初始化码本内存
    index.codebook.resize(M * Ks * index.subdim);
    index.codes.resize(base_number * M);

    // 对每个子空间进行 KMeans
    #pragma omp parallel for
    for (size_t m = 0; m < M; ++m) {
        // 提取该子空间所有子向量到连续数组（方便访问）
        std::vector<float> subspace(base_number * index.subdim);
        for (size_t i = 0; i < base_number; ++i) {
            std::memcpy(subspace.data() + i * index.subdim,
                        base + i * vecdim + m * index.subdim,
                        index.subdim * sizeof(float));
        }

        // 随机初始化 Ks 个中心（从数据中采样）
        std::vector<float> centroids(Ks * index.subdim);
        std::mt19937 rng(42 + m);
        std::uniform_int_distribution<size_t> dist(0, base_number - 1);
        for (size_t c = 0; c < Ks; ++c) {
            size_t idx = dist(rng);
            std::memcpy(centroids.data() + c * index.subdim,
                        subspace.data() + idx * index.subdim,
                        index.subdim * sizeof(float));
        }

        // Lloyd 迭代（固定 15 次）
        std::vector<size_t> assign(base_number);
        std::vector<size_t> cnt(Ks, 0);
        for (int iter = 0; iter < 15; ++iter) {
            // 分配步骤：计算每个向量到所有中心的距离，选最近
            #pragma omp parallel for
            for (size_t i = 0; i < base_number; ++i) {
                const float* subvec = subspace.data() + i * index.subdim;
                float best_dot = -1e30f;
                size_t best_c = 0;
                for (size_t c = 0; c < Ks; ++c) {
                    float dot = simd_dot_product(subvec,
                                                centroids.data() + c * index.subdim,
                                                index.subdim);
                    if (dot > best_dot) {
                        best_dot = dot;
                        best_c = c;
                    }
                }
                assign[i] = best_c;
            }

            // 更新步骤
            std::fill(cnt.begin(), cnt.end(), 0);
            std::vector<float> new_centroids(Ks * index.subdim, 0.0f);
            for (size_t i = 0; i < base_number; ++i) {
                size_t c = assign[i];
                cnt[c]++;
                const float* src = subspace.data() + i * index.subdim;
                float* dst = new_centroids.data() + c * index.subdim;
                for (size_t d = 0; d < index.subdim; ++d)
                    dst[d] += src[d];
            }
            for (size_t c = 0; c < Ks; ++c) {
                if (cnt[c] > 0) {
                    float inv = 1.0f / cnt[c];
                    for (size_t d = 0; d < index.subdim; ++d)
                        centroids[c * index.subdim + d] =
                            new_centroids[c * index.subdim + d] * inv;
                }
            }
        }

        // 保存该子空间码本
        std::memcpy(index.codebook.data() + m * Ks * index.subdim,
                    centroids.data(),
                    Ks * index.subdim * sizeof(float));

        // 量化 base 向量在该子空间的编码
        for (size_t i = 0; i < base_number; ++i) {
            const float* subvec = subspace.data() + i * index.subdim;
            float best_dot = -1e30f;
            uint8_t best_c = 0;
            for (size_t c = 0; c < Ks; ++c) {
                float dot = simd_dot_product(subvec,
                                            centroids.data() + c * index.subdim,
                                            index.subdim);
                if (dot > best_dot) {
                    best_dot = dot;
                    best_c = static_cast<uint8_t>(c);
                }
            }
            index.codes[i * M + m] = best_c;
        }
    }
}

// 计算单个子空间的 LUT（m 子空间，query 指向该子空间起始地址）
void compute_lut_subspace(const float* query_sub,
                          const float* codebook_sub,  // Ks * subdim
                          size_t Ks, size_t subdim,
                          float* lut) {
    #pragma omp parallel for
    for (size_t c = 0; c < Ks; ++c) {
        const float* center = codebook_sub + c * subdim;
        lut[c] = simd_dot_product(query_sub, center, subdim);
    }
}

// 粗排：使用 pq 编码筛选 top-p 候选
std::vector<uint32_t> pq_coarse_search(const PQIndex& index,
                                       const std::vector<std::vector<float>>& LUTs,
                                       size_t base_number, size_t p) {
    // LUTs: M 个子空间，每个大小为 Ks 的 float 数组
    struct Candidate {
        float dist;
        uint32_t id;
        bool operator<(const Candidate& o) const { return dist < o.dist; }
    };
    std::priority_queue<Candidate> heap;  // 最大堆，保留距离最小的 p 个

    const size_t M = index.M;
    const uint8_t* codes = index.codes.data();

    #pragma omp parallel
    {
        std::priority_queue<Candidate> local_heap;
        #pragma omp for nowait
        for (size_t i = 0; i < base_number; i += 4) {
            // 预取下一批编码
            if (i + 7 < base_number) {
                __builtin_prefetch(codes + (i + 4) * M, 0, 3);
            }

            // 加载 4 个向量的编码（共 16 字节）
            uint32_t idx_start = i * M;
            uint8x16_t codes_vec = vld1q_u8(codes + idx_start);

            // 提取每个向量的 4 个索引
            uint8_t idx[4][4];
            // 将 16 个 uint8 重新解释为 4 个 uint32，便于提取（小端）
            uint32_t raw[4];
            vst1q_u8(reinterpret_cast<uint8_t*>(raw), codes_vec);
            for (int j = 0; j < 4; ++j) {
                uint32_t val = raw[j];
                idx[j][0] = val & 0xFF;
                idx[j][1] = (val >> 8) & 0xFF;
                idx[j][2] = (val >> 16) & 0xFF;
                idx[j][3] = (val >> 24) & 0xFF;
            }

            // 对 4 个向量分别查表累加
            float dots[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (size_t m = 0; m < M; ++m) {
                const float* lut = LUTs[m].data();
                for (int j = 0; j < 4; ++j) {
                    dots[j] += lut[idx[j][m]];
                }
            }

            // 插入局部堆
            for (int j = 0; j < 4 && (i + j) < base_number; ++j) {
                float dist = 1.0f - dots[j];
                if (local_heap.size() < p) {
                    local_heap.push({dist, static_cast<uint32_t>(i + j)});
                } else if (dist < local_heap.top().dist) {
                    local_heap.pop();
                    local_heap.push({dist, static_cast<uint32_t>(i + j)});
                }
            }
        }
        // 合并局部堆
        #pragma omp critical
        {
            while (!local_heap.empty()) {
                auto c = local_heap.top();
                local_heap.pop();
                if (heap.size() < p) {
                    heap.push(c);
                } else if (c.dist < heap.top().dist) {
                    heap.pop();
                    heap.push(c);
                }
            }
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(heap.size());
    while (!heap.empty()) {
        candidates.push_back(heap.top().id);
        heap.pop();
    }
    std::reverse(candidates.begin(), candidates.end());  // 距离递增
    return candidates;
}

// 对指定候选集计算精确内积距离，返回 top-k（最大堆，距离小的优先）
std::priority_queue<std::pair<float, uint32_t>>
simd_rerank(const float* base, const float* query,
            const std::vector<uint32_t>& candidates,
            size_t vecdim, size_t k) {
    assert(vecdim == 96);
    std::priority_queue<std::pair<float, uint32_t>> q;

    // 预加载查询向量到寄存器
    const float32x4_t q0  = vld1q_f32(query);
    const float32x4_t q1  = vld1q_f32(query + 4);
    const float32x4_t q2  = vld1q_f32(query + 8);
    const float32x4_t q3  = vld1q_f32(query + 12);
    const float32x4_t q4  = vld1q_f32(query + 16);
    const float32x4_t q5  = vld1q_f32(query + 20);
    const float32x4_t q6  = vld1q_f32(query + 24);
    const float32x4_t q7  = vld1q_f32(query + 28);
    const float32x4_t q8  = vld1q_f32(query + 32);
    const float32x4_t q9  = vld1q_f32(query + 36);
    const float32x4_t q10 = vld1q_f32(query + 40);
    const float32x4_t q11 = vld1q_f32(query + 44);
    const float32x4_t q12 = vld1q_f32(query + 48);
    const float32x4_t q13 = vld1q_f32(query + 52);
    const float32x4_t q14 = vld1q_f32(query + 56);
    const float32x4_t q15 = vld1q_f32(query + 60);
    const float32x4_t q16 = vld1q_f32(query + 64);
    const float32x4_t q17 = vld1q_f32(query + 68);
    const float32x4_t q18 = vld1q_f32(query + 72);
    const float32x4_t q19 = vld1q_f32(query + 76);
    const float32x4_t q20 = vld1q_f32(query + 80);
    const float32x4_t q21 = vld1q_f32(query + 84);
    const float32x4_t q22 = vld1q_f32(query + 88);
    const float32x4_t q23 = vld1q_f32(query + 92);

    for (size_t idx : candidates) {
        const float* cur = base + idx * vecdim;
        float32x4_t prod0  = vmulq_f32(vld1q_f32(cur),        q0);
        float32x4_t prod1  = vmulq_f32(vld1q_f32(cur + 4),    q1);
        float32x4_t prod2  = vmulq_f32(vld1q_f32(cur + 8),    q2);
        float32x4_t prod3  = vmulq_f32(vld1q_f32(cur + 12),   q3);
        float32x4_t prod4  = vmulq_f32(vld1q_f32(cur + 16),   q4);
        float32x4_t prod5  = vmulq_f32(vld1q_f32(cur + 20),   q5);
        float32x4_t prod6  = vmulq_f32(vld1q_f32(cur + 24),   q6);
        float32x4_t prod7  = vmulq_f32(vld1q_f32(cur + 28),   q7);
        float32x4_t prod8  = vmulq_f32(vld1q_f32(cur + 32),   q8);
        float32x4_t prod9  = vmulq_f32(vld1q_f32(cur + 36),   q9);
        float32x4_t prod10 = vmulq_f32(vld1q_f32(cur + 40),   q10);
        float32x4_t prod11 = vmulq_f32(vld1q_f32(cur + 44),   q11);
        float32x4_t prod12 = vmulq_f32(vld1q_f32(cur + 48),   q12);
        float32x4_t prod13 = vmulq_f32(vld1q_f32(cur + 52),   q13);
        float32x4_t prod14 = vmulq_f32(vld1q_f32(cur + 56),   q14);
        float32x4_t prod15 = vmulq_f32(vld1q_f32(cur + 60),   q15);
        float32x4_t prod16 = vmulq_f32(vld1q_f32(cur + 64),   q16);
        float32x4_t prod17 = vmulq_f32(vld1q_f32(cur + 68),   q17);
        float32x4_t prod18 = vmulq_f32(vld1q_f32(cur + 72),   q18);
        float32x4_t prod19 = vmulq_f32(vld1q_f32(cur + 76),   q19);
        float32x4_t prod20 = vmulq_f32(vld1q_f32(cur + 80),   q20);
        float32x4_t prod21 = vmulq_f32(vld1q_f32(cur + 84),   q21);
        float32x4_t prod22 = vmulq_f32(vld1q_f32(cur + 88),   q22);
        float32x4_t prod23 = vmulq_f32(vld1q_f32(cur + 92),   q23);

        float32x4_t s01   = vaddq_f32(prod0,  prod1);
        float32x4_t s23   = vaddq_f32(prod2,  prod3);
        float32x4_t s45   = vaddq_f32(prod4,  prod5);
        float32x4_t s67   = vaddq_f32(prod6,  prod7);
        float32x4_t s89   = vaddq_f32(prod8,  prod9);
        float32x4_t s1011 = vaddq_f32(prod10, prod11);
        float32x4_t s1213 = vaddq_f32(prod12, prod13);
        float32x4_t s1415 = vaddq_f32(prod14, prod15);
        float32x4_t s1617 = vaddq_f32(prod16, prod17);
        float32x4_t s1819 = vaddq_f32(prod18, prod19);
        float32x4_t s2021 = vaddq_f32(prod20, prod21);
        float32x4_t s2223 = vaddq_f32(prod22, prod23);

        float32x4_t s0123 = vaddq_f32(s01, s23);
        float32x4_t s4567 = vaddq_f32(s45, s67);
        float32x4_t s891011 = vaddq_f32(s89, s1011);
        float32x4_t s12131415 = vaddq_f32(s1213, s1415);
        float32x4_t s16171819 = vaddq_f32(s1617, s1819);
        float32x4_t s20212223 = vaddq_f32(s2021, s2223);

        float32x4_t s0_7   = vaddq_f32(s0123, s4567);
        float32x4_t s8_15  = vaddq_f32(s891011, s12131415);
        float32x4_t s16_23 = vaddq_f32(s16171819, s20212223);

        float32x4_t sum_total = vaddq_f32(vaddq_f32(s0_7, s8_15), s16_23);
        float32x2_t sum_low = vadd_f32(vget_high_f32(sum_total), vget_low_f32(sum_total));
        float dot = vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);

        float dis = 1.0f - dot;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(idx)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(idx)});
            q.pop();
        }
    }
    return q;
}

// 完整 PQ 搜索接口：返回类型与原 sq_search 一致
std::priority_queue<std::pair<float, uint32_t>>
pq_search(const PQIndex& index, const float* base, const float* query,
          size_t base_number, size_t vecdim, size_t k, size_t p) {
    // 1. 构建 LUTs
    std::vector<std::vector<float>> LUTs(index.M,
                                         std::vector<float>(index.Ks));
    for (size_t m = 0; m < index.M; ++m) {
        const float* query_sub = query + m * index.subdim;
        const float* codebook_sub = index.codebook.data() + m * index.Ks * index.subdim;
        compute_lut_subspace(query_sub, codebook_sub,
                             index.Ks, index.subdim, LUTs[m].data());
    }

    // 2. 粗排获得 Top‑p 候选
    std::vector<uint32_t> candidates = pq_coarse_search(index, LUTs, base_number, p);

    // 3. 精排获得最终 Top‑k
    return simd_rerank(base, query, candidates, vecdim, k);
}

