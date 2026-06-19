#ifndef IVF_SIMD_H
#define IVF_SIMD_H

#include <vector>
#include <queue>
#include <algorithm>
#include <random>
#include <limits>
#include <arm_neon.h>
#include <omp.h>  // 引入 OpenMP 头文件

// IVF 索引结构体
struct IVFIndex {
    int nlist; // 聚类中心数量
    size_t vecdim; // 向量维度
    float* centroids; // 聚类中心向量 (nlist * vecdim)
    std::vector<std::vector<uint32_t>> inverted_lists; // 倒排列表
    IVFIndex() : nlist(0), vecdim(0), centroids(nullptr) {}
};

// ARM NEON 优化的内积计算函数
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

// 构建 IVF 索引 (简化的 K-Means 聚类)
inline void build_ivf_index(float* base, size_t base_number, size_t vecdim, IVFIndex& ivf_idx, int nlist = 1024) {
    ivf_idx.nlist = nlist;
    ivf_idx.vecdim = vecdim;
    ivf_idx.centroids = new float[nlist * vecdim];
    ivf_idx.inverted_lists.resize(nlist);

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
                if (ip > max_ip) {
                    max_ip = ip;
                    best_list = c;
                }
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

    for (size_t i = 0; i < base_number; ++i) {
        float max_ip = -std::numeric_limits<float>::max();
        int best_list = 0;
        for (int c = 0; c < nlist; ++c) {
            float ip = compute_ip_neon(base + i * vecdim, ivf_idx.centroids + c * vecdim, vecdim);
            if (ip > max_ip) {
                max_ip = ip;
                best_list = c;
            }
        }
        ivf_idx.inverted_lists[best_list].push_back(i);
    }
}

// 修改IVF-SIMD搜索函数，实现查询内并行（簇划分并行）
inline std::priority_queue<std::pair<float, uint32_t>> ivf_simd_search(
    const IVFIndex& ivf_idx, 
    float* base, 
    const float* query, 
    size_t base_number, 
    size_t vecdim, 
    size_t k, 
    size_t nprobe) {
    
    // 1. 粗排：计算 query 到所有聚类中心的内积
    // 实验指导书指出粗排计算量较小，多线程可能因为同步开销导致负优化，因此此处保持串行
    std::vector<std::pair<float, uint32_t>> centroid_dists(ivf_idx.nlist);
    for (int i = 0; i < ivf_idx.nlist; ++i) {
        float ip = compute_ip_neon(query, ivf_idx.centroids + i * vecdim, vecdim);
        centroid_dists[i] = {ip, i};
    }
    
    // 选出内积最大的 nprobe 个聚类中心 (降序排列)
    std::partial_sort(centroid_dists.begin(), centroid_dists.begin() + nprobe, centroid_dists.end(),
        [](const std::pair<float, uint32_t>& a, const std::pair<float, uint32_t>& b) {
            return a.first > b.first;
        });

    // 2. 精排：多线程扫描选中的倒排列表
    // 提前分配线程局部存储空间，避免锁竞争
    int max_threads = omp_get_max_threads();
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_top_k_queues(max_threads);

    // 使用动态调度，因为不同倒排列表的长度可能非常不均匀
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t i = 0; i < nprobe; ++i) {
        int tid = omp_get_thread_num();
        uint32_t list_id = centroid_dists[i].second;
        const auto& inv_list = ivf_idx.inverted_lists[list_id];
        
        // 每个线程维护自己的局部 top-k 结果
        for (uint32_t base_id : inv_list) {
            float ip = compute_ip_neon(query, base + base_id * vecdim, vecdim);
            float neg_ip = -ip; // 存入负数以模拟小顶堆逻辑
            
            if (local_top_k_queues[tid].size() < k) {
                local_top_k_queues[tid].push({neg_ip, base_id});
            } else {
                if (neg_ip < local_top_k_queues[tid].top().first) {
                    local_top_k_queues[tid].pop();
                    local_top_k_queues[tid].push({neg_ip, base_id});
                }
            }
        }
    }
    
    // 3. 合并各线程局部结果到全局 top-k
    std::priority_queue<std::pair<float, uint32_t>> global_top_k;
    for (int t = 0; t < max_threads; ++t) {
        while (!local_top_k_queues[t].empty()) {
            float neg_ip = local_top_k_queues[t].top().first;
            uint32_t base_id = local_top_k_queues[t].top().second;
            local_top_k_queues[t].pop();
            
            if (global_top_k.size() < k) {
                global_top_k.push({neg_ip, base_id});
            } else {
                if (neg_ip < global_top_k.top().first) {
                    global_top_k.pop();
                    global_top_k.push({neg_ip, base_id});
                }
            }
        }
    }
    return global_top_k;
}

#endif // IVF_SIMD_H
