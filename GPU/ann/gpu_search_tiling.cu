#include "gpu_search_tiling.cuh"
#include <cuda_runtime.h>
#include <climits>
#include <iostream>

static float* d_base = nullptr;
static size_t d_base_number = 0;
static size_t d_vecdim = 0;

void init_gpu_base(float* base, size_t base_number, size_t vecdim) {
    d_base_number = base_number;
    d_vecdim = vecdim;
    cudaMalloc(&d_base, base_number * vecdim * sizeof(float));
    cudaMemcpy(d_base, base, base_number * vecdim * sizeof(float), cudaMemcpyHostToDevice);
}

void free_gpu_base() {
    if (d_base) cudaFree(d_base);
    d_base = nullptr;
}

// ==========================================
// 优化1：矩阵乘法 Tiling (共享内存分块)
// ==========================================
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(const float* B, const float* Q, float* D, int N, int d, int m) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; // Base向量索引 (0 ~ N-1)
    int col = blockIdx.y * blockDim.y + threadIdx.y; // Query向量索引 (0 ~ m-1)
    
    if (row < N && col < m) {
        float dot = 0.0f;
        
        // 按维度 d 切分 Tile
        for (int p = 0; p < (d + TILE_SIZE - 1) / TILE_SIZE; ++p) {
            __shared__ float tile_B[TILE_SIZE][TILE_SIZE];
            __shared__ float tile_Q[TILE_SIZE][TILE_SIZE];
            
            // 协作加载 Base 的 Tile
            int b_col = p * TILE_SIZE + threadIdx.y;
            if (b_col < d) {
                tile_B[threadIdx.x][threadIdx.y] = B[row * d + b_col];
            } else {
                tile_B[threadIdx.x][threadIdx.y] = 0.0f;
            }

            // 协作加载 Query 的 Tile
            int q_row = p * TILE_SIZE + threadIdx.x;
            if (q_row < d) {
                tile_Q[threadIdx.x][threadIdx.y] = Q[col * d + q_row];
            } else {
                tile_Q[threadIdx.x][threadIdx.y] = 0.0f;
            }
            
            __syncthreads(); // 等待 Tile 加载完毕

            // 在共享内存中进行乘累加
            for (int i = 0; i < TILE_SIZE; ++i) {
                dot += tile_B[threadIdx.x][i] * tile_Q[i][threadIdx.y];
            }
            __syncthreads(); // 确保计算完成后再加载下一个 Tile
        }
        D[col * N + row] = 1.0f - dot;
    }
}

// ==========================================
// 优化2：各线程维护 Top-p 并行归约
// ==========================================
// 指导书提及"各线程维护top-p"，这里设 p=10 (等于 k)
const int TOP_P = 10; 

__global__ void topk_opt_kernel(float* D, uint32_t* out_indices, int N, int m, int k) {
    int col = blockIdx.x; // 每个Block处理一个查询
    if (col >= m) return;

    const int THREADS = 256;
    // 每个线程维护一个大小为 TOP_P 的局部数组
    float local_vals[TOP_P];
    int local_idxs[TOP_P];
    for(int i=0; i<TOP_P; i++) { local_vals[i] = 1e20f; local_idxs[i] = -1; }

    // 1. 每个线程扫描距离数组的一段，维护局部的 Top-P
    for (int i = threadIdx.x; i < N; i += THREADS) {
        float val = D[col * N + i];
        // 如果比当前局部最大值小，则替换并插入排序
        if (val < local_vals[TOP_P - 1]) {
            local_vals[TOP_P - 1] = val;
            local_idxs[TOP_P - 1] = i;
            for(int j = TOP_P - 1; j > 0; j--) {
                if (local_vals[j] < local_vals[j-1]) {
                    float tv = local_vals[j]; local_vals[j] = local_vals[j-1]; local_vals[j-1] = tv;
                    int ti = local_idxs[j]; local_idxs[j] = local_idxs[j-1]; local_idxs[j-1] = ti;
                } else break;
            }
        }
    }

    // 2. 将线程的局部结果写入共享内存
    __shared__ float s_vals[THREADS * TOP_P];
    __shared__ int s_idxs[THREADS * TOP_P];
    for(int i=0; i<TOP_P; i++) {
        s_vals[threadIdx.x * TOP_P + i] = local_vals[i];
        s_idxs[threadIdx.x * TOP_P + i] = local_idxs[i];
    }
    __syncthreads();

    // 3. Block内归约：让线程0从 256*10 个候选中挑出最终的 Top-K
    if (threadIdx.x == 0) {
        for(int kv=0; kv<k; ++kv) {
            float min_v = 1e20f;
            int min_pos = -1;
            for(int i=0; i<THREADS * TOP_P; ++i) {
                if(s_vals[i] < min_v) {
                    min_v = s_vals[i];
                    min_pos = i;
                }
            }
            out_indices[col * k + kv] = s_idxs[min_pos];
            s_vals[min_pos] = 1e20f; // 找到一个后置为无穷大，寻找下一个
        }
    }
}

std::vector<std::priority_queue<std::pair<float, uint32_t>>> gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size) {
    
    float* d_query;
    cudaMalloc(&d_query, batch_size * vecdim * sizeof(float));
    cudaMemcpy(d_query, query_batch, batch_size * vecdim * sizeof(float), cudaMemcpyHostToDevice);

    float* d_dist;
    cudaMalloc(&d_dist, batch_size * base_number * sizeof(float));

    // ====== CUDA Event 计时器初始化 ======
    cudaEvent_t start, stop_matmul, stop_topk;
    cudaEventCreate(&start);
    cudaEventCreate(&stop_matmul);
    cudaEventCreate(&stop_topk);
    float time_matmul, time_topk;

    cudaEventRecord(start, 0);
    // ====================================

    // 启动 Tiling 矩阵乘法
    dim3 block_mat(TILE_SIZE, TILE_SIZE);
    dim3 grid_mat((base_number + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    matmul_tiled_kernel<<<grid_mat, block_mat>>>(d_base, d_query, d_dist, base_number, vecdim, batch_size);

    // ====== 记录矩阵乘法结束时间 ======
    cudaEventRecord(stop_matmul, 0);
    // ====================================

    uint32_t* d_indices;
    cudaMalloc(&d_indices, batch_size * k * sizeof(uint32_t));

    // 启动 Top-p 归约 Kernel
    int threads_topk = 256;
    topk_opt_kernel<<<batch_size, threads_topk>>>(d_dist, d_indices, base_number, batch_size, k);

    // ====== 记录 Top-K 结束时间并同步 ======
    cudaEventRecord(stop_topk, 0);
    cudaEventSynchronize(stop_topk); // 等待所有 Kernel 执行完毕

    cudaEventElapsedTime(&time_matmul, start, stop_matmul);
    cudaEventElapsedTime(&time_topk, stop_matmul, stop_topk);
    
    // 输出每个 Kernel 的单条查询平均耗时 (转换为 us)
    std::cerr << "[Profile] Matmul: " << time_matmul * 1000 / batch_size << " us/query | "
              << "Top-K: " << time_topk * 1000 / batch_size << " us/query\n";
    // ====================================

    uint32_t* h_indices = new uint32_t[batch_size * k];
    cudaMemcpy(h_indices, d_indices, batch_size * k * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    std::vector<std::priority_queue<std::pair<float, uint32_t>>> results;
    for (size_t i = 0; i < batch_size; ++i) {
        std::priority_queue<std::pair<float, uint32_t>> q;
        for (size_t j = 0; j < k; ++j) {
            uint32_t idx = h_indices[i * k + j];
            q.push({0.0f, idx}); 
        }
        results.push_back(q);
    }

    delete[] h_indices;
    cudaFree(d_query);
    cudaFree(d_dist);
    cudaFree(d_indices);

    cudaEventDestroy(start);
    cudaEventDestroy(stop_matmul);
    cudaEventDestroy(stop_topk);

    return results;
}
