#pragma once
#include <queue>
#include <vector>
#include <utility>
#include <cstdint>

// 初始化IVF索引：在CPU执行K-Means，重排数据并上传GPU
void init_ivf_gpu(float* base, size_t base_number, size_t vecdim, size_t nlist = 100);

// IVF批量搜索
std::vector<std::priority_queue<std::pair<float, uint32_t>>> ivf_gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size, size_t nprobe = 10);

    // 新增：Batch分组矩阵乘法搜索
std::vector<std::priority_queue<std::pair<float, uint32_t>>> grouped_gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size, size_t nprobe);

// 释放GPU显存
void free_ivf_gpu();
