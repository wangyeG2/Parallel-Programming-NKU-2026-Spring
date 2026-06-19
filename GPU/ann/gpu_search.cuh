#pragma once
#include <queue>
#include <vector>
#include <utility>
#include <cstdint>

void init_gpu_base(float* base, size_t base_number, size_t vecdim);
void free_gpu_base();
std::vector<std::priority_queue<std::pair<float, uint32_t>>> gpu_search_batch(
    float* query_batch, size_t base_number, size_t vecdim, size_t k, size_t batch_size);
