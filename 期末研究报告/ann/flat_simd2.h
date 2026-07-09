#pragma once
#include <queue>
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>
#include <pthread.h>
#include <cassert>
#include <vector>
#include <algorithm>

// 定义线程参数结构体
typedef struct {
    const float* base;
    const float* query;
    size_t start_idx;
    size_t end_idx;
    size_t vecdim;
    size_t k;
    // 每个线程维护自己的局部优先队列，避免全局加锁
    std::priority_queue<std::pair<float, uint32_t>> local_q;
} ThreadArg;

// 线程执行函数：Base划分 + 局部SIMD计算 + 局部Top-K
void* flat_search_thread_func(void* arg) {
    ThreadArg* t_arg = (ThreadArg*)arg;
    
    const float* base = t_arg->base;
    const float* query = t_arg->query;
    size_t vecdim = t_arg->vecdim;
    size_t k = t_arg->k;

    // 将Query加载到NEON寄存器（与原代码一致，避免循环内重复加载）
    const float32x4_t q0 = vld1q_f32(query);
    const float32x4_t q1 = vld1q_f32(query + 4);
    const float32x4_t q2 = vld1q_f32(query + 8);
    const float32x4_t q3 = vld1q_f32(query + 12);
    const float32x4_t q4 = vld1q_f32(query + 16);
    const float32x4_t q5 = vld1q_f32(query + 20);
    const float32x4_t q6 = vld1q_f32(query + 24);
    const float32x4_t q7 = vld1q_f32(query + 28);
    const float32x4_t q8 = vld1q_f32(query + 32);
    const float32x4_t q9 = vld1q_f32(query + 36);
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

    // 遍历分配给该线程的Base向量子集
    for (size_t i = t_arg->start_idx; i < t_arg->end_idx; ++i) {
        const float* cur_base = base + i * vecdim;
        
        // 软件预取（距离3个迭代，可根据CPU L2 Cache大小调整）
        if (i + 3 < t_arg->end_idx) {
            __builtin_prefetch(base + (i + 3) * vecdim, 0, 3);
        }

        // NEON SIMD 内积计算（与原代码一致）
        float32x4_t prod0 = vmulq_f32(vld1q_f32(cur_base), q0);
        float32x4_t prod1 = vmulq_f32(vld1q_f32(cur_base + 4), q1);
        float32x4_t prod2 = vmulq_f32(vld1q_f32(cur_base + 8), q2);
        float32x4_t prod3 = vmulq_f32(vld1q_f32(cur_base + 12), q3);
        float32x4_t prod4 = vmulq_f32(vld1q_f32(cur_base + 16), q4);
        float32x4_t prod5 = vmulq_f32(vld1q_f32(cur_base + 20), q5);
        float32x4_t prod6 = vmulq_f32(vld1q_f32(cur_base + 24), q6);
        float32x4_t prod7 = vmulq_f32(vld1q_f32(cur_base + 28), q7);
        float32x4_t prod8 = vmulq_f32(vld1q_f32(cur_base + 32), q8);
        float32x4_t prod9 = vmulq_f32(vld1q_f32(cur_base + 36), q9);
        float32x4_t prod10 = vmulq_f32(vld1q_f32(cur_base + 40), q10);
        float32x4_t prod11 = vmulq_f32(vld1q_f32(cur_base + 44), q11);
        float32x4_t prod12 = vmulq_f32(vld1q_f32(cur_base + 48), q12);
        float32x4_t prod13 = vmulq_f32(vld1q_f32(cur_base + 52), q13);
        float32x4_t prod14 = vmulq_f32(vld1q_f32(cur_base + 56), q14);
        float32x4_t prod15 = vmulq_f32(vld1q_f32(cur_base + 60), q15);
        float32x4_t prod16 = vmulq_f32(vld1q_f32(cur_base + 64), q16);
        float32x4_t prod17 = vmulq_f32(vld1q_f32(cur_base + 68), q17);
        float32x4_t prod18 = vmulq_f32(vld1q_f32(cur_base + 72), q18);
        float32x4_t prod19 = vmulq_f32(vld1q_f32(cur_base + 76), q19);
        float32x4_t prod20 = vmulq_f32(vld1q_f32(cur_base + 80), q20);
        float32x4_t prod21 = vmulq_f32(vld1q_f32(cur_base + 84), q21);
        float32x4_t prod22 = vmulq_f32(vld1q_f32(cur_base + 88), q22);
        float32x4_t prod23 = vmulq_f32(vld1q_f32(cur_base + 92), q23);

        // 树状归约求和
        float32x4_t s01 = vaddq_f32(prod0, prod1);
        float32x4_t s23 = vaddq_f32(prod2, prod3);
        float32x4_t s45 = vaddq_f32(prod4, prod5);
        float32x4_t s67 = vaddq_f32(prod6, prod7);
        float32x4_t s89 = vaddq_f32(prod8, prod9);
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

        float32x4_t s0_7 = vaddq_f32(s0123, s4567);
        float32x4_t s8_15 = vaddq_f32(s891011, s12131415);
        float32x4_t s16_23 = vaddq_f32(s16171819, s20212223);

        float32x4_t sum_total = vaddq_f32(vaddq_f32(s0_7, s8_15), s16_23);
        float32x2_t sum_low = vadd_f32(vget_high_f32(sum_total), vget_low_f32(sum_total));
        float dot_product = vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);
        
        float dis = 1.0f - dot_product;

        // 写入局部优先队列（无需加锁）
        if (t_arg->local_q.size() < k) {
            t_arg->local_q.push({dis, i});
        } else if (dis < t_arg->local_q.top().first) {
            t_arg->local_q.push({dis, i});
            t_arg->local_q.pop();
        }
    }
    return nullptr;
}

// 主搜索函数：Pthread多线程驱动 + 全局Top-K归约
inline std::priority_queue<std::pair<float, uint32_t>> pthread_simd_flat_search(
    const float* base, const float* query, size_t base_number, size_t vecdim, size_t k, int num_threads) {
    
    assert(vecdim >= 96 && vecdim % 8 == 0);
    
    std::vector<pthread_t> threads(num_threads);
    std::vector<ThreadArg> args(num_threads);

    size_t chunk_size = base_number / num_threads;
    size_t remainder = base_number % num_threads;
    size_t current_start = 0;

    // 1. 划分任务并创建线程
    for (int i = 0; i < num_threads; ++i) {
        size_t current_chunk = chunk_size + (i < remainder ? 1 : 0);
        args[i].base = base;
        args[i].query = query;
        args[i].start_idx = current_start;
        args[i].end_idx = current_start + current_chunk;
        args[i].vecdim = vecdim;
        args[i].k = k;
        
        // 注意：局部优先队列默认构造为空，无需显式初始化
        pthread_create(&threads[i], nullptr, flat_search_thread_func, &args[i]);
        current_start += current_chunk;
    }

    // 2. 等待所有线程完成
    for (int i = 0; i < num_threads; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // 3. 全局Top-K归约：合并所有线程的局部优先队列
    std::priority_queue<std::pair<float, uint32_t>> global_q;
    for (int i = 0; i < num_threads; ++i) {
        while (!args[i].local_q.empty()) {
            auto elem = args[i].local_q.top();
            args[i].local_q.pop(); // 弹出以释放内存
            
            if (global_q.size() < k) {
                global_q.push(elem);
            } else if (elem.first < global_q.top().first) {
                global_q.push(elem);
                global_q.pop();
            }
        }
    }

    return global_q;
}
