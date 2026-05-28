// flat_simd_omp.h
#pragma once
#include <queue>
#include <cstddef>
#include <cstdint>
#include <arm_neon.h>
#include <omp.h>
#include <cassert>
#include <vector>

inline std::priority_queue<std::pair<float, uint32_t>> flat_simd_omp_search(
    const float* base, 
    const float* query, 
    size_t base_number, 
    size_t vecdim, 
    size_t k, 
    int num_threads,size_t local_p) 
{
    assert(vecdim >= 96 && vecdim % 8 == 0);
    omp_set_num_threads(num_threads);

    // 每个线程维护局部优先队列，避免加锁
    std::vector<std::priority_queue<std::pair<float, uint32_t>>> local_queues(num_threads);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        
        // 将Query加载到NEON寄存器（每个线程加载一次，避免重复访存）
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

        // 使用 schedule(runtime) 允许在运行时通过 omp_set_schedule 控制调度策略
        #pragma omp for schedule(runtime)
        for (size_t i = 0; i < base_number; ++i) {
            const float* cur_base = base + i * vecdim;

            // 软件预取
            if (i + 3 < base_number) {
                __builtin_prefetch(base + (i + 3) * vecdim, 0, 3);
            }

            // NEON SIMD 内积计算
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
            if (local_queues[tid].size() < local_p) { 
                local_queues[tid].push({dis, i});
            } else if (dis < local_queues[tid].top().first) {
                local_queues[tid].push({dis, i});
                local_queues[tid].pop();
            }
        }
    }

    // 全局Top-K归约：合并所有线程的局部优先队列
    std::priority_queue<std::pair<float, uint32_t>> global_q;
    for (int i = 0; i < num_threads; ++i) {
        while (!local_queues[i].empty()) {
            auto elem = local_queues[i].top();
            local_queues[i].pop();
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
