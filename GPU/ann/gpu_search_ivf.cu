#include "gpu_search_ivf.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <omp.h>
#include <algorithm> // for std::sort
#include <set>       // for std::set

// 全局GPU显存指针
static float* d_base_T = nullptr;      // 转置后的Base数据 (dim x N)
static float* d_centroids = nullptr;   // 聚类中心 (nlist x dim)
static int* d_cluster_offsets = nullptr; // 每个簇在d_base_T中的起始偏移 (nlist+1)
static int* d_original_indices = nullptr; // 新增：记录重排后位置对应的原始base索引
static int* h_original_indices_map = nullptr;
static size_t d_nlist = 0;

// ---------------- CPU端 K-Means 构建IVF索引 ----------------
void init_ivf_gpu(float* base, size_t base_number, size_t vecdim, size_t nlist) {
    d_nlist = nlist;
    float* centroids = new float[nlist * vecdim];
    int* cluster_id = new int[base_number];
    
    // 1. 随机初始化聚类中心
    srand(42);
    for (size_t i = 0; i < nlist; ++i) {
        size_t idx = rand() % base_number;
        memcpy(centroids + i * vecdim, base + idx * vecdim, vecdim * sizeof(float));
    }

    // 2. 简易K-Means迭代 (10次)
    size_t* cluster_sizes = new size_t[nlist];
    for (int iter = 0; iter < 10; ++iter) {
        #pragma omp parallel for
        for (size_t i = 0; i < base_number; ++i) {
            float max_dot = -1e20f;
            int best_c = 0;
            for (size_t c = 0; c < nlist; ++c) {
                float dot = 0;
                for (size_t j = 0; j < vecdim; ++j) {
                    dot += base[i * vecdim + j] * centroids[c * vecdim + j];
                }
                if (dot > max_dot) { max_dot = dot; best_c = c; }
            }
            cluster_id[i] = best_c;
        }

        memset(cluster_sizes, 0, nlist * sizeof(size_t));
        memset(centroids, 0, nlist * vecdim * sizeof(float));
        for (size_t i = 0; i < base_number; ++i) {
            int c = cluster_id[i];
            cluster_sizes[c]++;
            for (size_t j = 0; j < vecdim; ++j) {
                centroids[c * vecdim + j] += base[i * vecdim + j];
            }
        }
        for (size_t c = 0; c < nlist; ++c) {
            if (cluster_sizes[c] > 0) {
                for (size_t j = 0; j < vecdim; ++j) centroids[c * vecdim + j] /= cluster_sizes[c];
            } else {
                // 处理空簇：随机分配一个点作为新中心，防止该簇永远不被访问
                size_t idx = rand() % base_number;
                memcpy(centroids + c * vecdim, base + idx * vecdim, vecdim * sizeof(float));
            }
        }
    }

    // 3. 数据重排与转置，并记录原始索引映射
    int* h_offsets = new int[nlist + 1]();
    for (size_t i = 0; i < base_number; ++i) h_offsets[cluster_id[i] + 1]++;
    for (size_t c = 0; c < nlist; ++c) h_offsets[c + 1] += h_offsets[c];

    float* h_base_T = new float[base_number * vecdim];
    int* h_original_indices = new int[base_number]; // 记录映射关系
    int* current_pos = new int[nlist]();
    
    for (size_t i = 0; i < base_number; ++i) {
        int c = cluster_id[i];
        int pos = h_offsets[c] + current_pos[c]++;
        h_original_indices[pos] = i; // pos位置存的是原来的第i个向量
        for (size_t j = 0; j < vecdim; ++j) {
            h_base_T[j * base_number + pos] = base[i * vecdim + j];
        }
    }

    // 4. 上传至GPU
    cudaMalloc(&d_base_T, base_number * vecdim * sizeof(float));
    cudaMemcpy(d_base_T, h_base_T, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_centroids, nlist * vecdim * sizeof(float));
    cudaMemcpy(d_centroids, centroids, nlist * vecdim * sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMalloc(&d_cluster_offsets, (nlist + 1) * sizeof(int));
    cudaMemcpy(d_cluster_offsets, h_offsets, (nlist + 1) * sizeof(int), cudaMemcpyHostToDevice);

    // 上传原索引映射表
    cudaMalloc(&d_original_indices, base_number * sizeof(int));
    cudaMemcpy(d_original_indices, h_original_indices, base_number * sizeof(int), cudaMemcpyHostToDevice);

    h_original_indices_map = h_original_indices; // 保存到全局供分组映射使用
    
    delete[] centroids; delete[] cluster_id; delete[] cluster_sizes;
    delete[] h_offsets; delete[] h_base_T; delete[] current_pos; 
}


void free_ivf_gpu() {
    if (d_base_T) cudaFree(d_base_T);
    if (d_centroids) cudaFree(d_centroids);
    if (d_cluster_offsets) cudaFree(d_cluster_offsets);
    if (d_original_indices) cudaFree(d_original_indices);
}

// ---------------- GPU Kernel 1: 寻找 nprobe 个最近簇 ----------------
__global__ void find_nprobe_kernel(const float* Q, const float* C, int* probes, int N, int d, int m, int nlist, int nprobe) {
    int q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= m) return;
    
    const float* q = Q + q_idx * d;
    float dists[256]; // 假设nlist <= 256
    for (int c = 0; c < nlist; ++c) {
        float dot = 0;
        for (int j = 0; j < d; ++j) dot += q[j] * C[c * d + j];
        dists[c] = 1.0f - dot;
    }
    
    for (int p = 0; p < nprobe; ++p) {
        float min_val = 1e20f; int min_idx = -1;
        for (int c = 0; c < nlist; ++c) {
            if (dists[c] < min_val) { min_val = dists[c]; min_idx = c; }
        }
        probes[q_idx * nprobe + p] = min_idx;
        dists[min_idx] = 1e20f;
    }
}

// ---------------- GPU Kernel 2: IVF 细粒度搜索 ----------------
__global__ void ivf_search_kernel(const float* Q, const float* B_T, const int* offsets, const int* probes, const int* original_indices, int* out_idx, int N, int d, int m, int k, int nprobe) {
    int q_idx = blockIdx.x;
    if (q_idx >= m) return;
    
    const float* q = Q + q_idx * d;
    const int* my_probes = probes + q_idx * nprobe;

    // 将query向量加载到共享内存
    extern __shared__ float s_q[];
    if (threadIdx.x < d) s_q[threadIdx.x] = q[threadIdx.x];
    __syncthreads();

    // 每个线程维护自己的局部Top-K
    const int LOCAL_K = 10;
    float local_val[LOCAL_K];
    int local_idx[LOCAL_K];
    for(int i=0; i<LOCAL_K; ++i) { local_val[i] = 1e20f; local_idx[i] = -1; }

    // 遍历分配给该查询的 nprobe 个簇
    for (int p = 0; p < nprobe; ++p) {
        int c = my_probes[p];
        int start = offsets[c];
        int end = offsets[c + 1];
        
        for (int i = start + threadIdx.x; i < end; i += blockDim.x) {
            float dot = 0;
            for (int j = 0; j < d; ++j) {
                dot += s_q[j] * B_T[j * N + i];
            }
            float dist = 1.0f - dot;
            
            if (dist < local_val[LOCAL_K - 1]) {
                local_val[LOCAL_K - 1] = dist;
                local_idx[LOCAL_K - 1] = i; // i是重排后的位置
                for(int t = LOCAL_K - 2; t >= 0; --t) {
                    if (local_val[t+1] < local_val[t]) {
                        float tv = local_val[t]; local_val[t] = local_val[t+1]; local_val[t+1] = tv;
                        int ti = local_idx[t]; local_idx[t] = local_idx[t+1]; local_idx[t+1] = ti;
                    } else {
                        break; // 提前结束冒泡
                    }
                }
            }
        }
    }

    // 块内归约
    __shared__ float block_val[256 * LOCAL_K];
    __shared__ int block_idx[256 * LOCAL_K];
    for(int i=0; i<LOCAL_K; ++i) {
        block_val[threadIdx.x * LOCAL_K + i] = local_val[i];
        block_idx[threadIdx.x * LOCAL_K + i] = local_idx[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float final_val[10];
        int final_idx[10];
        for(int i=0; i<10; ++i) final_val[i] = 1e20f;
        
        for (int i = 0; i < blockDim.x * LOCAL_K; ++i) {
            if (block_val[i] < final_val[9]) {
                final_val[9] = block_val[i];
                final_idx[9] = block_idx[i];
                for(int t = 8; t >= 0; --t) {
                    if (final_val[t+1] < final_val[t]) {
                        float tv = final_val[t]; final_val[t] = final_val[t+1]; final_val[t+1] = tv;
                        int ti = final_idx[t]; final_idx[t] = final_idx[t+1]; final_idx[t+1] = ti;
                    } else {
                        break;
                    }
                }
            }
        }
        
        // 核心修复：通过 original_indices 数组将重排位置映射回原始base索引
        for(int i=0; i<k; ++i) {
            if (final_idx[i] != -1) {
                out_idx[q_idx * k + i] = original_indices[final_idx[i]];
            } else {
                out_idx[q_idx * k + i] = 0;
            }
        }
    }
}

// ---------------- IVF 批量搜索接口 ----------------
std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size, size_t nprobe) {
    
    float* d_query;
    cudaMalloc(&d_query, batch_size * vecdim * sizeof(float));
    cudaMemcpy(d_query, query_batch, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    int* d_probes;
    cudaMalloc(&d_probes, batch_size * nprobe * sizeof(int));

    // 1. 寻找 nprobe 个最近簇
    int threads_probe = 64;
    int blocks_probe = (batch_size + threads_probe - 1) / threads_probe;
    find_nprobe_kernel<<<blocks_probe, threads_probe>>>(d_query, d_centroids, d_probes, base_number, vecdim, batch_size, d_nlist, nprobe);

    // 2. 在选定簇内搜索 Top-K
    uint32_t* d_indices;
    cudaMalloc(&d_indices, batch_size * k * sizeof(uint32_t));
    
    int threads_search = 256;
    int smem_size = vecdim * sizeof(float);
    
    // 传入 d_original_indices 进行映射还原
    ivf_search_kernel<<<batch_size, threads_search, smem_size>>>(
        d_query, d_base_T, d_cluster_offsets, d_probes, d_original_indices, (int*)d_indices, base_number, vecdim, batch_size, k, nprobe);

    // 3. 结果回传与格式转换
    uint32_t* h_indices = new uint32_t[batch_size * k];
    cudaMemcpy(h_indices, d_indices, batch_size * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results;
    for (size_t i = 0; i < batch_size; ++i) {
        std::priority_queue<std::pair<float, uint32_t>> q;
        for (size_t j = 0; j < k; ++j) {
            q.push({0.0f, h_indices[i * k + j]});
        }
        results.push_back(q);
    }

    delete[] h_indices;
    cudaFree(d_query);
    cudaFree(d_probes);
    cudaFree(d_indices);
    return results;
}
// ================= Batch分组策略实现 =================

struct QueryMeta {
    int original_idx;
    int first_probe;
};

// Kernel 1: 提取向量到连续内存
__global__ void gather_kernel(const float* B_T, float* group_base, const int* vec_indices, int U_g, int d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < U_g) {
        int original_pos = vec_indices[idx];
        for (int j = 0; j < d; ++j) {
            group_base[idx * d + j] = B_T[j * N + original_pos];
        }
    }
}

// Kernel 2: 分组矩阵乘法 + Mask掩码
__global__ void grouped_matmul_kernel(
    const float* group_base, const float* Q, float* dist_out, 
    const int* group_query_probes, const int* vec_cluster_ids, 
    int U_g, int d, int group_size, int nprobe) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < U_g && col < group_size) {
        int c_id = vec_cluster_ids[row];
        
        // 检查当前向量所在的簇是否在当前Query的 nprobe 列表中
        bool valid = false;
        for (int p = 0; p < nprobe; ++p) {
            if (group_query_probes[col * nprobe + p] == c_id) {
                valid = true;
                break;
            }
        }
        
        if (valid) {
            float dot = 0;
            for (int j = 0; j < d; ++j) {
                dot += group_base[row * d + j] * Q[col * d + j];
            }
            dist_out[col * U_g + row] = 1.0f - dot;
        } else {
            dist_out[col * U_g + row] = 1e20f; // 掩码掉无效计算
        }
    }
}

// Kernel 3: 组内 Top-K 寻找
__global__ void grouped_topk_kernel(float* D, uint32_t* out_indices, int U_g, int group_size, int k) {
    int col = blockIdx.x;
    if (col >= group_size) return;
    
    __shared__ float s_min_val[256];
    __shared__ int s_min_idx[256];
    
    for (int kv = 0; kv < k; ++kv) {
        int tid = threadIdx.x;
        float local_min_val = 1e20f;
        int local_min_idx = -1;
        
        for (int i = tid; i < U_g; i += blockDim.x) {
            float val = D[col * U_g + i];
            if (val < local_min_val) {
                local_min_val = val;
                local_min_idx = i;
            }
        }
        s_min_val[tid] = local_min_val;
        s_min_idx[tid] = local_min_idx;
        __syncthreads();
        
        for (int s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                if (s_min_val[tid + s] < s_min_val[tid]) {
                    s_min_val[tid] = s_min_val[tid + s];
                    s_min_idx[tid] = s_min_idx[tid + s];
                }
            }
            __syncthreads();
        }
        
        if (tid == 0) {
            out_indices[col * k + kv] = s_min_idx[0];
            D[col * U_g + s_min_idx[0]] = 1e20f;
        }
        __syncthreads();
    }
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> grouped_gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size, size_t nprobe) {
    
    float* d_query;
    cudaMalloc(&d_query, batch_size * vecdim * sizeof(float));
    cudaMemcpy(d_query, query_batch, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    int* d_probes;
    cudaMalloc(&d_probes, batch_size * nprobe * sizeof(int));

    // 1. 找出所有Query的 nprobe 个簇
    int threads_probe = 64;
    int blocks_probe = (batch_size + threads_probe - 1) / threads_probe;
    find_nprobe_kernel<<<blocks_probe, threads_probe>>>(d_query, d_centroids, d_probes, base_number, vecdim, batch_size, d_nlist, nprobe);

    // 2. CPU端获取 probes 进行排序分组
    int* h_probes = new int[batch_size * nprobe];
    cudaMemcpy(h_probes, d_probes, batch_size * nprobe * sizeof(int), cudaMemcpyDeviceToHost);

    std::vector<QueryMeta> query_metas(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        query_metas[i].original_idx = i;
        query_metas[i].first_probe = h_probes[i * nprobe];
    }
    // 按照第一个簇的ID排序，使得相邻Query的簇重合度极高
    std::sort(query_metas.begin(), query_metas.end(), [](const QueryMeta& a, const QueryMeta& b) {
        return a.first_probe < b.first_probe;
    });

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> final_results(batch_size);

    // 3. 分组处理 (以10个Query为一组)
    const int group_size = 10;
    int num_groups = (batch_size + group_size - 1) / group_size;

    int* h_offsets = new int[d_nlist + 1];
    cudaMemcpy(h_offsets, d_cluster_offsets, (d_nlist + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int g = 0; g < num_groups; ++g) {
        int current_group_size = std::min(group_size, (int)batch_size - g * group_size);
        if (current_group_size <= 0) break;

        // 求该组所有Query的 nprobe 簇的并集
        std::set<int> union_probes_set;
        for (int i = 0; i < current_group_size; ++i) {
            int q_idx = query_metas[g * group_size + i].original_idx;
            for (int p = 0; p < nprobe; ++p) {
                union_probes_set.insert(h_probes[q_idx * nprobe + p]);
            }
        }

        std::vector<int> union_probes(union_probes_set.begin(), union_probes_set.end());
        int num_union = union_probes.size();

        // 构造该组的向量列表和簇ID列表
        std::vector<int> h_vec_indices;
        std::vector<int> h_vec_cluster_ids;
        for (int c : union_probes) {
            int start = h_offsets[c];
            int end = h_offsets[c + 1];
            for (int i = start; i < end; ++i) {
                h_vec_indices.push_back(i);
                h_vec_cluster_ids.push_back(c);
            }
        }
        int U_g = h_vec_indices.size();

        // --- 上传该组数据到GPU ---
        int* d_vec_indices;
        cudaMalloc(&d_vec_indices, U_g * sizeof(int));
        cudaMemcpy(d_vec_indices, h_vec_indices.data(), U_g * sizeof(int), cudaMemcpyHostToDevice);

        int* d_vec_cluster_ids;
        cudaMalloc(&d_vec_cluster_ids, U_g * sizeof(int));
        cudaMemcpy(d_vec_cluster_ids, h_vec_cluster_ids.data(), U_g * sizeof(int), cudaMemcpyHostToDevice);

        int* d_group_query_probes;
        cudaMalloc(&d_group_query_probes, current_group_size * nprobe * sizeof(int));
        for (int i = 0; i < current_group_size; ++i) {
            int q_idx = query_metas[g * group_size + i].original_idx;
            cudaMemcpy(d_group_query_probes + i * nprobe, h_probes + q_idx * nprobe, nprobe * sizeof(int), cudaMemcpyHostToDevice);
        }

        float* d_group_query;
        cudaMalloc(&d_group_query, current_group_size * vecdim * sizeof(float));
        for (int i = 0; i < current_group_size; ++i) {
            int q_idx = query_metas[g * group_size + i].original_idx;
            cudaMemcpy(d_group_query + i * vecdim, d_query + q_idx * vecdim, vecdim * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        // --- Kernel 1: Gather 向量并转为连续矩阵 ---
        float* d_group_base;
        cudaMalloc(&d_group_base, U_g * vecdim * sizeof(float));
        int threads_gather = 256;
        int blocks_gather = (U_g + threads_gather - 1) / threads_gather;
        gather_kernel<<<blocks_gather, threads_gather>>>(d_base_T, d_group_base, d_vec_indices, U_g, vecdim, base_number);

        // --- Kernel 2: 分组矩阵乘法 + Mask ---
        float* d_group_dist;
        cudaMalloc(&d_group_dist, current_group_size * U_g * sizeof(float));
        dim3 block_mat(16, 16);
        dim3 grid_mat((U_g + 15) / 16, (current_group_size + 15) / 16);
        grouped_matmul_kernel<<<grid_mat, block_mat>>>(
            d_group_base, d_group_query, d_group_dist, 
            d_group_query_probes, d_vec_cluster_ids, 
            U_g, vecdim, current_group_size, nprobe);

        // --- Kernel 3: 组内 Top-K ---
        uint32_t* d_group_indices;
        cudaMalloc(&d_group_indices, current_group_size * k * sizeof(uint32_t));
        int threads_topk = 256;
        grouped_topk_kernel<<<current_group_size, threads_topk>>>(d_group_dist, d_group_indices, U_g, current_group_size, k);

        // --- 结果映射回原全局索引 ---
        uint32_t* h_group_indices = new uint32_t[current_group_size * k];
        cudaMemcpy(h_group_indices, d_group_indices, current_group_size * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);

        for (int i = 0; i < current_group_size; ++i) {
            int q_idx = query_metas[g * group_size + i].original_idx;
            std::priority_queue<std::pair<float, uint32_t>> q;
            for (int j = 0; j < k; ++j) {
                int local_idx = h_group_indices[i * k + j];
                if (local_idx >= 0 && local_idx < U_g) {
                    int rearranged_pos = h_vec_indices[local_idx];
                    int original_base_id = h_original_indices_map[rearranged_pos];
                    q.push({0.0f, original_base_id});
                }
            }
            final_results[q_idx] = q;
        }

        delete[] h_group_indices;
        cudaFree(d_vec_indices);
        cudaFree(d_vec_cluster_ids);
        cudaFree(d_group_query_probes);
        cudaFree(d_group_query);
        cudaFree(d_group_base);
        cudaFree(d_group_dist);
        cudaFree(d_group_indices);
    }

    delete[] h_probes;
    delete[] h_offsets;
    cudaFree(d_query);
    cudaFree(d_probes);

    return final_results;
}
