#ifndef IVF_PQ_SIMD_H
#define IVF_PQ_SIMD_H

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <limits>
#include <cstdint>
#include <cstring>
#include <arm_neon.h>
#include <omp.h> // 引入 OpenMP 头文件

// ---------------- 数据结构定义（保持不变） ----------------
struct PQCodebook {
    int M;
    int Ks;
    int sub_dim;
    float* centroids;
};

struct IVFPQIndex {
    int nlist;
    size_t vecdim;
    float* ivf_centroids;
    std::vector<std::vector<uint32_t>> inverted_lists;
    std::vector<std::vector<uint8_t>> inverted_lists_pq_codes;
    PQCodebook pq;
    uint8_t* pq_codes;
};

// ---------------- NEON 辅助函数（4路 FMA 展开，保持不变） ----------------
inline float compute_ip_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec0 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec1 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec2 = vdupq_n_f32(0.0f);
    float32x4_t sum_vec3 = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 16 <= dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);
        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);
        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);
        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum_vec0 = vfmaq_f32(sum_vec0, a0, b0);
        sum_vec1 = vfmaq_f32(sum_vec1, a1, b1);
        sum_vec2 = vfmaq_f32(sum_vec2, a2, b2);
        sum_vec3 = vfmaq_f32(sum_vec3, a3, b3);
    }

    sum_vec0 = vaddq_f32(sum_vec0, sum_vec1);
    sum_vec2 = vaddq_f32(sum_vec2, sum_vec3);
    sum_vec0 = vaddq_f32(sum_vec0, sum_vec2);

    float res = vaddvq_f32(sum_vec0);

    for (; i < dim; ++i) {
        res += a[i] * b[i];
    }

    return res;
}

// ---------------- 索引构建（保持不变） ----------------
inline void build_ivf_pq_index(float* base, size_t base_number, size_t vecdim, IVFPQIndex& idx, int nlist = 256, int M = 4, int Ks = 256) {
    idx.nlist = nlist;
    idx.vecdim = vecdim;
    idx.pq.M = M;
    idx.pq.Ks = Ks;
    idx.pq.sub_dim = vecdim / M;

    idx.ivf_centroids = new float[nlist * vecdim];
    idx.pq.centroids = new float[M * Ks * idx.pq.sub_dim];
    idx.pq_codes = new uint8_t[base_number * M];
    idx.inverted_lists.resize(nlist);

    // 1. IVF KMeans 初始化与迭代
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, base_number - 1);

    for (int i = 0; i < nlist; ++i) {
        int src_id = dis(gen);
        memcpy(idx.ivf_centroids + i * vecdim, base + src_id * vecdim, vecdim * sizeof(float));
    }

    std::vector<int> assignments(base_number);
    for (int iter = 0; iter < 10; ++iter) {
        for (size_t i = 0; i < base_number; ++i) {
            float max_ip = -std::numeric_limits<float>::max();
            int best_list = 0;
            for (int c = 0; c < nlist; ++c) {
                float ip = compute_ip_neon(base + i * vecdim, idx.ivf_centroids + c * vecdim, vecdim);
                if (ip > max_ip) {
                    max_ip = ip;
                    best_list = c;
                }
            }
            assignments[i] = best_list;
        }

        std::vector<float> new_cents(nlist * vecdim, 0.0f);
        std::vector<int> cluster_sizes(nlist, 0);
        for (size_t i = 0; i < base_number; ++i) {
            int c = assignments[i];
            cluster_sizes[c]++;
            for (size_t d = 0; d < vecdim; ++d) {
                new_cents[c * vecdim + d] += base[i * vecdim + d];
            }
        }
        for (int c = 0; c < nlist; ++c) {
            if (cluster_sizes[c] > 0) {
                for (size_t d = 0; d < vecdim; ++d) {
                    idx.ivf_centroids[c * vecdim + d] = new_cents[c * vecdim + d] / cluster_sizes[c];
                }
            }
        }
    }

    // 2. PQ 码本构建
    for (int m = 0; m < M; ++m) {
        int sub_d = idx.pq.sub_dim;
        float* sub_centroids = idx.pq.centroids + m * Ks * sub_d;

        for (int k = 0; k < Ks; ++k) {
            int src_id = dis(gen);
            memcpy(sub_centroids + k * sub_d, base + src_id * vecdim + m * sub_d, sub_d * sizeof(float));
        }

        for (int iter = 0; iter < 5; ++iter) {
            std::vector<int> sub_assignments(base_number);
            for (size_t i = 0; i < base_number; ++i) {
                float max_ip = -std::numeric_limits<float>::max();
                int best_k = 0;
                for (int k = 0; k < Ks; ++k) {
                    float ip = compute_ip_neon(base + i * vecdim + m * sub_d, sub_centroids + k * sub_d, sub_d);
                    if (ip > max_ip) {
                        max_ip = ip;
                        best_k = k;
                    }
                }
                sub_assignments[i] = best_k;
            }

            std::vector<float> new_sub_cents(Ks * sub_d, 0.0f);
            std::vector<int> sub_sizes(Ks, 0);
            for (size_t i = 0; i < base_number; ++i) {
                int k = sub_assignments[i];
                sub_sizes[k]++;
                for (int d = 0; d < sub_d; ++d) {
                    new_sub_cents[k * sub_d + d] += base[i * vecdim + m * sub_d + d];
                }
            }
            for (int k = 0; k < Ks; ++k) {
                if (sub_sizes[k] > 0) {
                    for (int d = 0; d < sub_d; ++d) {
                        sub_centroids[k * sub_d + d] = new_sub_cents[k * sub_d + d] / sub_sizes[k];
                    }
                }
            }
        }
    }

    // 3. PQ 编码与倒排列表分配
    for (size_t i = 0; i < base_number; ++i) {
        float max_ip = -std::numeric_limits<float>::max();
        int best_list = 0;
        for (int c = 0; c < nlist; ++c) {
            float ip = compute_ip_neon(base + i * vecdim, idx.ivf_centroids + c * vecdim, vecdim);
            if (ip > max_ip) {
                max_ip = ip;
                best_list = c;
            }
        }
        idx.inverted_lists[best_list].push_back(i);

        for (int m = 0; m < M; ++m) {
            float max_sub_ip = -std::numeric_limits<float>::max();
            int best_k = 0;
            for (int k = 0; k < Ks; ++k) {
                float ip = compute_ip_neon(base + i * vecdim + m * idx.pq.sub_dim, idx.pq.centroids + (m * Ks + k) * idx.pq.sub_dim, idx.pq.sub_dim);
                if (ip > max_sub_ip) {
                    max_sub_ip = ip;
                    best_k = k;
                }
            }
            idx.pq_codes[i * M + m] = static_cast<uint8_t>(best_k);
        }
    }

    // SoA 数据重排
    idx.inverted_lists_pq_codes.resize(nlist);
    for (int c = 0; c < nlist; ++c) {
        size_t list_size = idx.inverted_lists[c].size();
        idx.inverted_lists_pq_codes[c].resize(list_size * M);
        for (size_t j = 0; j < list_size; ++j) {
            uint32_t id = idx.inverted_lists[c][j];
            for (int m = 0; m < M; ++m) {
                idx.inverted_lists_pq_codes[c][m * list_size + j] = idx.pq_codes[id * M + m];
            }
        }
    }
}

// ---------------- 优化后的 IVF-PQ 搜索（多线程版） ----------------
inline std::priority_queue<std::pair<float, uint32_t>> ivf_pq_simd_search(
    const IVFPQIndex& idx, float* base, const float* query, size_t base_number, size_t vecdim, size_t k, size_t nprobe, size_t p = 100) 
{
    int M = idx.pq.M;
    int Ks = idx.pq.Ks;
    int sub_dim = idx.pq.sub_dim;

    // 1. 粗排（计算量较小，保持串行，避免线程开销）
    std::vector<std::pair<float, uint32_t>> centroid_dists(idx.nlist);
    for (int i = 0; i < idx.nlist; ++i) {
        float ip = compute_ip_neon(query, idx.ivf_centroids + i * vecdim, vecdim);
        centroid_dists[i] = {ip, i};
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end(), 
                      [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first > b.first; });

    // 2. LUT 构建（并行优化：按子空间M划分）
    std::vector<float> lut_float(M * Ks);
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        for (int k_idx = 0; k_idx < Ks; ++k_idx) {
            lut_float[m * Ks + k_idx] = compute_ip_neon(query + m * sub_dim, idx.pq.centroids + (m * Ks + k_idx) * sub_dim, sub_dim);
        }
    }

    // 3. LUT 8-bit 量化（串行）
    float min_ip = std::numeric_limits<float>::max();
    float max_ip = -std::numeric_limits<float>::max();
    for (int i = 0; i < M * Ks; ++i) {
        if (lut_float[i] < min_ip) min_ip = lut_float[i];
        if (lut_float[i] > max_ip) max_ip = lut_float[i];
    }

    std::vector<uint8_t> lut_8bit(M * Ks);
    float scale = (max_ip > min_ip) ? (255.0f / (max_ip - min_ip)) : 0.0f;
    for (int i = 0; i < M * Ks; ++i) {
        lut_8bit[i] = static_cast<uint8_t>((lut_float[i] - min_ip) * scale);
    }

    // ================= 核心优化 1：数组收集 + 多线程局部存储 =================
    struct Candidate {
        uint16_t score;
        uint32_t id;
    };

    int max_threads = omp_get_max_threads();
    std::vector<std::vector<Candidate>> local_candidates(max_threads);
    for (int t = 0; t < max_threads; ++t) {
        local_candidates[t].reserve(512); // 预留空间避免频繁扩容
    }

    // 4. 查表累加：多线程并行 + 动态调度解决倒排列表长度不均的问题
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 1)
        for (size_t i = 0; i < nprobe; ++i) {
            uint32_t list_id = centroid_dists[i].second;
            const std::vector<uint32_t>& inv_list = idx.inverted_lists[list_id];
            const std::vector<uint8_t>& list_codes = idx.inverted_lists_pq_codes[list_id];
            size_t list_size = inv_list.size();
            size_t j = 0;

            // 主循环：每次处理 16 个向量
            for (; j + 16 <= list_size; j += 16) {
                uint16x8_t sum_vec0 = vdupq_n_u16(0);
                uint16x8_t sum_vec1 = vdupq_n_u16(0);
                for (int m = 0; m < M; ++m) {
                    const uint8_t* codes_ptr = list_codes.data() + m * list_size + j;
                    __builtin_prefetch(codes_ptr + 64, 0, 3);
                    uint8x16_t codes_vec = vld1q_u8(codes_ptr);

                    uint8_t lut_vals[16];
                    uint8x8_t codes_low = vget_low_u8(codes_vec);
                    uint8x8_t codes_high = vget_high_u8(codes_vec);
                    lut_vals[0] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 0)];
                    lut_vals[1] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 1)];
                    lut_vals[2] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 2)];
                    lut_vals[3] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 3)];
                    lut_vals[4] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 4)];
                    lut_vals[5] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 5)];
                    lut_vals[6] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 6)];
                    lut_vals[7] = lut_8bit[m * Ks + vget_lane_u8(codes_low, 7)];
                    lut_vals[8] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 0)];
                    lut_vals[9] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 1)];
                    lut_vals[10] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 2)];
                    lut_vals[11] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 3)];
                    lut_vals[12] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 4)];
                    lut_vals[13] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 5)];
                    lut_vals[14] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 6)];
                    lut_vals[15] = lut_8bit[m * Ks + vget_lane_u8(codes_high, 7)];

                    uint8x16_t l_vec = vld1q_u8(lut_vals);
                    uint16x8_t l_low = vmovl_u8(vget_low_u8(l_vec));
                    uint16x8_t l_high = vmovl_u8(vget_high_u8(l_vec));
                    sum_vec0 = vaddq_u16(sum_vec0, l_low);
                    sum_vec1 = vaddq_u16(sum_vec1, l_high);
                }
                uint16_t tmp_sums[16];
                vst1q_u16(tmp_sums, sum_vec0);
                vst1q_u16(tmp_sums + 8, sum_vec1);

                for (int t = 0; t < 16; ++t) {
                    local_candidates[tid].push_back({tmp_sums[t], inv_list[j + t]});
                }
            }

            // 尾部：8 向量批量
            for (; j + 8 <= list_size; j += 8) {
                uint16x8_t sum_vec = vdupq_n_u16(0);
                for (int m = 0; m < M; ++m) {
                    const uint8_t* codes_ptr = list_codes.data() + m * list_size + j;
                    uint8x8_t codes_vec = vld1_u8(codes_ptr);
                    uint8_t lut_vals[8];
                    lut_vals[0] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 0)];
                    lut_vals[1] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 1)];
                    lut_vals[2] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 2)];
                    lut_vals[3] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 3)];
                    lut_vals[4] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 4)];
                    lut_vals[5] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 5)];
                    lut_vals[6] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 6)];
                    lut_vals[7] = lut_8bit[m * Ks + vget_lane_u8(codes_vec, 7)];

                    uint8x8_t l_vec = vld1_u8(lut_vals);
                    sum_vec = vaddq_u16(sum_vec, vmovl_u8(l_vec));
                }
                uint16_t tmp_sums[8];
                vst1q_u16(tmp_sums, sum_vec);
                for (int t = 0; t < 8; ++t) {
                    local_candidates[tid].push_back({tmp_sums[t], inv_list[j + t]});
                }
            }

            // 剩余标量尾部
            for (; j < list_size; ++j) {
                uint16_t ip = 0;
                for (int m = 0; m < M; ++m) {
                    ip += lut_8bit[m * Ks + list_codes[m * list_size + j]];
                }
                local_candidates[tid].push_back({ip, inv_list[j]});
            }
        }
    }

    // 合并各线程的局部候选
    size_t total_candidates = 0;
    for (int t = 0; t < max_threads; ++t) {
        total_candidates += local_candidates[t].size();
    }
    std::vector<Candidate> all_candidates;
    all_candidates.reserve(total_candidates);
    for (int t = 0; t < max_threads; ++t) {
        all_candidates.insert(all_candidates.end(), local_candidates[t].begin(), local_candidates[t].end());
    }

    // ================= 核心优化 2：nth_element 选 Top-p =================
    size_t actual_p = std::min(p, all_candidates.size());
    std::nth_element(all_candidates.begin(), all_candidates.begin() + actual_p, all_candidates.end(), 
                     [](const Candidate& a, const Candidate& b) { return a.score > b.score; });

    std::vector<uint32_t> top_p_ids;
    top_p_ids.reserve(actual_p);
    for (size_t i = 0; i < actual_p; ++i) {
        top_p_ids.push_back(all_candidates[i].id);
    }

    // 5. 精排：多线程并行计算内积
    std::vector<float> exact_scores(actual_p);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < top_p_ids.size(); ++i) {
        uint32_t id = top_p_ids[i];
        exact_scores[i] = compute_ip_neon(query, base + (size_t)id * vecdim, vecdim);
    }

    // 串行构建 top-k 堆
    std::priority_queue<std::pair<float, uint32_t>> top_k_results;
    for (size_t i = 0; i < top_p_ids.size(); ++i) {
        float neg_ip = -exact_scores[i];
        if (top_k_results.size() < k) {
            top_k_results.push({neg_ip, top_p_ids[i]});
        } else if (neg_ip < top_k_results.top().first) {
            top_k_results.pop();
            top_k_results.push({neg_ip, top_p_ids[i]});
        }
    }

    return top_k_results;
}

#endif // IVF_PQ_SIMD_H
