#pragma once
#include <queue>
#include <immintrin.h> // AVX2 和 FMA 指令头文件

std::priority_queue<std::pair<float, uint32_t>> flat_search_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    
    // 循环遍历所有 base 向量
    for (int i = 0; i < base_number; ++i) {
        float dis = 0;
        __m256 vsum = _mm256_setzero_ps(); // 初始化 256 位累加器为 0
        
        int d = 0;
        // AVX2 主循环：每次处理 8 个 float
        for (; d + 8 <= vecdim; d += 8) {
            __m256 v_base = _mm256_loadu_ps(&base[d + i * vecdim]);
            __m256 v_query = _mm256_loadu_ps(&query[d]);
            // FMA 指令：vsum = v_base * v_query + vsum
            vsum = _mm256_fmadd_ps(v_base, v_query, vsum);
        }
        
        // 水平求和：将 256 位寄存器中的 8 个 float 累加到一个标量 dis 中
        // 提取低 128 位和高 128 位
        __m128 vlow = _mm256_castps256_ps128(vsum);
        __m128 vhigh = _mm256_extractf128_ps(vsum, 1);
        __m128 vsum128 = _mm_add_ps(vlow, vhigh);
        
        // 128 位内水平相加
        __m128 shuf = _mm_movehdup_ps(vsum128);
        __m128 sums = _mm_add_ps(vsum128, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        dis = _mm_cvtss_f32(sums);
        
        // 处理尾部不足 8 个的剩余元素
        for (; d < vecdim; ++d) {
            dis += base[d + i * vecdim] * query[d];
        }
        
        dis = 1 - dis;
        
        // 维护 Top-K 的优先队列
        if (q.size() < k) {
            q.push({dis, i});
        } else {
            if (dis < q.top().first) {
                q.push({dis, i});
                q.pop();
            }
        }
    }
    return q;
}
