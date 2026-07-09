#ifndef IVF_SIMD_MPI_H
#define IVF_SIMD_MPI_H

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <limits>
#include <arm_neon.h>
#include <mpi.h> // 引入 MPI 头文件

// IVF 索引结构体 (保持不变)
struct IVFIndex {
    int nlist;
    size_t vecdim;
    float* centroids;
    std::vector<std::vector<uint32_t>> inverted_lists;
    IVFIndex() : nlist(0), vecdim(0), centroids(nullptr) {}
};

// ARM NEON 优化的内积计算函数 (保持不变)
inline float compute_ip_neon(const float* a, const float* b, size_t dim) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t a_vec = vld1q_f32(a + i);
        float32x4_t b_vec = vld1q_f32(b + i);
        sum_vec = vmlaq_f32(sum_vec, a_vec, b_vec);
    }
    float res = vaddvq_f32(sum_vec);
    for (; i < dim; ++i) {
        res += a[i] * b[i];
    }
    return res;
}

// 构建 IVF 索引 (MPI 分布式版本)
inline void build_ivf_index_mpi(float* base, size_t base_number, size_t vecdim, IVFIndex& ivf_idx, int nlist, int rank, int size) {
    ivf_idx.nlist = nlist;
    ivf_idx.vecdim = vecdim;
    ivf_idx.centroids = new float[nlist * vecdim];
    ivf_idx.inverted_lists.resize(nlist);

    // 1. Rank 0 负责执行全局 K-Means 聚类，生成全局聚类中心
    if (rank == 0) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, base_number - 1);
        for (int i = 0; i < nlist; ++i) {
            int src_id = dis(gen);
            memcpy(ivf_idx.centroids + i * vecdim, base + src_id * vecdim, vecdim * sizeof(float));
        }
        std::vector<int> assignments(base_number);
        for (int iter = 0; iter < 10; ++iter) {
            for (size_t i = 0; i < base_number; ++i) {
                float max_ip = -std::numeric_limits<float>::max();
                int best_list = 0;
                for (int c = 0; c < nlist; ++c) {
                    float ip = compute_ip_neon(base + i * vecdim, ivf_idx.centroids + c * vecdim, vecdim);
                    if (ip > max_ip) { max_ip = ip; best_list = c; }
                }
                assignments[i] = best_list;
            }
            std::vector<float> new_centroids(nlist * vecdim, 0.0f);
            std::vector<int> cluster_sizes(nlist, 0);
            for (size_t i = 0; i < base_number; ++i) {
                int c = assignments[i];
                cluster_sizes[c]++;
                for (size_t d = 0; d < vecdim; ++d) {
                    new_centroids[c * vecdim + d] += base[i * vecdim + d];
                }
            }
            for (int c = 0; c < nlist; ++c) {
                if (cluster_sizes[c] > 0) {
                    for (size_t d = 0; d < vecdim; ++d) {
                        ivf_idx.centroids[c * vecdim + d] = new_centroids[c * vecdim + d] / cluster_sizes[c];
                    }
                }
            }
        }
    }

    // 2. 广播全局聚类中心给所有进程
    MPI_Bcast(ivf_idx.centroids, nlist * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 3. 数据划分：各进程只对自己负责的那部分 base 数据构建本地倒排列表
    size_t chunk = (base_number + size - 1) / size; // 每个进程处理的向量数(向上取整)
    size_t start_id = rank * chunk;
    size_t end_id = std::min(start_id + chunk, base_number);

    for (size_t i = start_id; i < end_id; ++i) {
        float max_ip = -std::numeric_limits<float>::max();
        int best_list = 0;
        for (int c = 0; c < nlist; ++c) {
            float ip = compute_ip_neon(base + i * vecdim, ivf_idx.centroids + c * vecdim, vecdim);
            if (ip > max_ip) { max_ip = ip; best_list = c; }
        }
        // 注意：这里存的是全局ID，因为最终返回的结果需要是全局唯一的
        ivf_idx.inverted_lists[best_list].push_back(i);
    }
}

// IVF-SIMD-MPI 搜索函数
inline std::priority_queue<std::pair<float, uint32_t>> ivf_simd_mpi_search(
    const IVFIndex& ivf_idx, float* base, const float* query, 
    size_t base_number, size_t vecdim, size_t k, size_t nprobe, int rank, int size) 
{
    // 1. 粗排：所有进程都有全局质心，独立计算 nprobe 个最近簇 (计算量小，无需并行)
    std::vector<std::pair<float, uint32_t>> centroid_dists(ivf_idx.nlist);
    for (int i = 0; i < ivf_idx.nlist; ++i) {
        float ip = compute_ip_neon(query, ivf_idx.centroids + i * vecdim, vecdim);
        centroid_dists[i] = {ip, i};
    }
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end(),
        [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) { return a.first > b.first; });

    // 2. 精排：各进程遍历自己维护的本地倒排列表，计算局部 Top-K
    std::priority_queue<std::pair<float, uint32_t>> local_top_k;
    for (size_t i = 0; i < nprobe; ++i) {
        uint32_t list_id = centroid_dists[i].second;
        const auto& inv_list = ivf_idx.inverted_lists[list_id];
        for (uint32_t base_id : inv_list) {
            float ip = compute_ip_neon(query, base + base_id * vecdim, vecdim);
            float neg_ip = -ip;
            if (local_top_k.size() < k) {
                local_top_k.push({neg_ip, base_id});
            } else {
                if (neg_ip < local_top_k.top().first) {
                    local_top_k.pop();
                    local_top_k.push({neg_ip, base_id});
                }
            }
        }
    }

    // 3. 全局 Top-K 合并 (踩分点：merge/reduce)
    // 将优先队列转换为定长数组，便于 MPI_Gather 通信
    std::vector<std::pair<float, uint32_t>> local_arr(k, {0.0f, UINT32_MAX});
    int idx = 0;
    while (!local_top_k.empty() && idx < k) {
        local_arr[idx] = local_top_k.top();
        local_top_k.pop();
        idx++;
    }

    std::vector<std::pair<float, uint32_t>> global_arr;
    if (rank == 0) {
        global_arr.resize(size * k);
    }

    // 收集所有进程的局部 Top-K 结果到 Rank 0
    MPI_Gather(local_arr.data(), k * sizeof(std::pair<float, uint32_t>), MPI_BYTE,
               global_arr.data(), k * sizeof(std::pair<float, uint32_t>), MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Rank 0 负责从所有收集到的结果中合并出最终的全局 Top-K
    std::priority_queue<std::pair<float, uint32_t>> global_top_k;
    if (rank == 0) {
        for (int p = 0; p < size; ++p) {
            for (int i = 0; i < k; ++i) {
                auto& pair = global_arr[p * k + i];
                if (pair.second != UINT32_MAX) { // 过滤掉补齐的无效数据
                    if (global_top_k.size() < k) {
                        global_top_k.push(pair);
                    } else if (pair.first < global_top_k.top().first) {
                        global_top_k.pop();
                        global_top_k.push(pair);
                    }
                }
            }
        }
    }
    return global_top_k;
}

#endif // IVF_SIMD_MPI_H
