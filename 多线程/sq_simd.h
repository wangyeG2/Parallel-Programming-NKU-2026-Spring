#pragma once

#include <vector>
#include <queue>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <arm_neon.h>

struct SQIndex {
    std::vector<uint8_t> codes;         
    std::vector<float>   scale;         
    std::vector<float>   offset;        
    size_t vecdim;
};

inline void build_sq_index(const float* base, size_t base_number,
                           size_t vecdim, SQIndex& idx) {
    idx.vecdim = vecdim;
    idx.scale.resize(vecdim);
    idx.offset.resize(vecdim);
    idx.codes.resize(base_number * vecdim);

    for (size_t d = 0; d < vecdim; ++d) {
        float min_v = base[d];
        float max_v = base[d];
        for (size_t i = 1; i < base_number; ++i) {
            float v = base[i * vecdim + d];
            if (v < min_v) min_v = v;
            if (v > max_v) max_v = v;
        }
        idx.offset[d] = min_v;
        float range = max_v - min_v;
        if (range < 1e-8f) range = 1e-8f;
        idx.scale[d] = range / 255.0f;
    }

    for (size_t i = 0; i < base_number; ++i) {
        for (size_t d = 0; d < vecdim; ++d) {
            float val = base[i * vecdim + d];
            int q = static_cast<int>(
                round((val - idx.offset[d]) / idx.scale[d]));
            if (q < 0) q = 0;
            if (q > 255) q = 255;
            idx.codes[i * vecdim + d] = static_cast<uint8_t>(q);
        }
    }
}

inline float ip_96_neon(const float* a, const float* b) {
    float32x4_t a0  = vld1q_f32(a);
    float32x4_t a1  = vld1q_f32(a+4);
    float32x4_t a2  = vld1q_f32(a+8);
    float32x4_t a3  = vld1q_f32(a+12);
    float32x4_t a4  = vld1q_f32(a+16);
    float32x4_t a5  = vld1q_f32(a+20);
    float32x4_t a6  = vld1q_f32(a+24);
    float32x4_t a7  = vld1q_f32(a+28);
    float32x4_t a8  = vld1q_f32(a+32);
    float32x4_t a9  = vld1q_f32(a+36);
    float32x4_t a10 = vld1q_f32(a+40);
    float32x4_t a11 = vld1q_f32(a+44);
    float32x4_t a12 = vld1q_f32(a+48);
    float32x4_t a13 = vld1q_f32(a+52);
    float32x4_t a14 = vld1q_f32(a+56);
    float32x4_t a15 = vld1q_f32(a+60);
    float32x4_t a16 = vld1q_f32(a+64);
    float32x4_t a17 = vld1q_f32(a+68);
    float32x4_t a18 = vld1q_f32(a+72);
    float32x4_t a19 = vld1q_f32(a+76);
    float32x4_t a20 = vld1q_f32(a+80);
    float32x4_t a21 = vld1q_f32(a+84);
    float32x4_t a22 = vld1q_f32(a+88);
    float32x4_t a23 = vld1q_f32(a+92);

    float32x4_t b0  = vld1q_f32(b);
    float32x4_t b1  = vld1q_f32(b+4);
    float32x4_t b2  = vld1q_f32(b+8);
    float32x4_t b3  = vld1q_f32(b+12);
    float32x4_t b4  = vld1q_f32(b+16);
    float32x4_t b5  = vld1q_f32(b+20);
    float32x4_t b6  = vld1q_f32(b+24);
    float32x4_t b7  = vld1q_f32(b+28);
    float32x4_t b8  = vld1q_f32(b+32);
    float32x4_t b9  = vld1q_f32(b+36);
    float32x4_t b10 = vld1q_f32(b+40);
    float32x4_t b11 = vld1q_f32(b+44);
    float32x4_t b12 = vld1q_f32(b+48);
    float32x4_t b13 = vld1q_f32(b+52);
    float32x4_t b14 = vld1q_f32(b+56);
    float32x4_t b15 = vld1q_f32(b+60);
    float32x4_t b16 = vld1q_f32(b+64);
    float32x4_t b17 = vld1q_f32(b+68);
    float32x4_t b18 = vld1q_f32(b+72);
    float32x4_t b19 = vld1q_f32(b+76);
    float32x4_t b20 = vld1q_f32(b+80);
    float32x4_t b21 = vld1q_f32(b+84);
    float32x4_t b22 = vld1q_f32(b+88);
    float32x4_t b23 = vld1q_f32(b+92);

    float32x4_t prod0  = vmulq_f32(a0,  b0);
    float32x4_t prod1  = vmulq_f32(a1,  b1);
    float32x4_t prod2  = vmulq_f32(a2,  b2);
    float32x4_t prod3  = vmulq_f32(a3,  b3);
    float32x4_t prod4  = vmulq_f32(a4,  b4);
    float32x4_t prod5  = vmulq_f32(a5,  b5);
    float32x4_t prod6  = vmulq_f32(a6,  b6);
    float32x4_t prod7  = vmulq_f32(a7,  b7);
    float32x4_t prod8  = vmulq_f32(a8,  b8);
    float32x4_t prod9  = vmulq_f32(a9,  b9);
    float32x4_t prod10 = vmulq_f32(a10, b10);
    float32x4_t prod11 = vmulq_f32(a11, b11);
    float32x4_t prod12 = vmulq_f32(a12, b12);
    float32x4_t prod13 = vmulq_f32(a13, b13);
    float32x4_t prod14 = vmulq_f32(a14, b14);
    float32x4_t prod15 = vmulq_f32(a15, b15);
    float32x4_t prod16 = vmulq_f32(a16, b16);
    float32x4_t prod17 = vmulq_f32(a17, b17);
    float32x4_t prod18 = vmulq_f32(a18, b18);
    float32x4_t prod19 = vmulq_f32(a19, b19);
    float32x4_t prod20 = vmulq_f32(a20, b20);
    float32x4_t prod21 = vmulq_f32(a21, b21);
    float32x4_t prod22 = vmulq_f32(a22, b22);
    float32x4_t prod23 = vmulq_f32(a23, b23);

    float32x4_t s01   = vaddq_f32(prod0,  prod1);
    float32x4_t s23   = vaddq_f32(prod2,  prod3);
    float32x4_t s45   = vaddq_f32(prod4,  prod5);
    float32x4_t s67   = vaddq_f32(prod6,  prod7);
    float32x4_t s89   = vaddq_f32(prod8,  prod9);
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

    float32x4_t s0_7   = vaddq_f32(s0123, s4567);
    float32x4_t s8_15  = vaddq_f32(s891011, s12131415);
    float32x4_t s16_23 = vaddq_f32(s16171819, s20212223);

    float32x4_t sum_total = vaddq_f32(vaddq_f32(s0_7, s8_15), s16_23);
    float32x2_t sum_low = vadd_f32(vget_high_f32(sum_total),
                                   vget_low_f32(sum_total));
    return vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);
}

static void prepare_query_weights(const float* query,
                                  const SQIndex& idx,
                                  int16_t* qw_int16) {
    for (size_t d = 0; d < idx.vecdim; ++d) {
        float w = query[d] * idx.scale[d];
        const float factor = 32768.0f;
        int val = (int)round(w * factor);
        if (val > 32767) val = 32767;
        if (val < -32768) val = -32768;
        qw_int16[d] = (int16_t)val;
    }
}
inline std::priority_queue<std::pair<float, uint32_t>>
sq_search(const SQIndex& idx, const float* base, const float* query,
          size_t base_number, size_t vecdim, size_t k, size_t p) {
    assert(vecdim == 96);

    int16_t qw[96] __attribute__((aligned(16)));
    prepare_query_weights(query, idx, qw);

    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>> coarse_heap;

    const uint8_t* codes_ptr = idx.codes.data();
    for (size_t i = 0; i < base_number; ++i) {
        const uint8_t* code = codes_ptr + i * vecdim;

        int32x4_t sum = vdupq_n_s32(0);

        for (int g = 0; g < 6; ++g) {
            uint8x16_t c = vld1q_u8(code + g * 16);

            int16x8_t c_low  = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(c)));
            int16x8_t c_high = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(c)));

            int16x8_t w_low  = vld1q_s16(qw + g * 16);
            int16x8_t w_high = vld1q_s16(qw + g * 16 + 8);

            sum = vmlal_s16(sum, vget_low_s16(c_low),  vget_low_s16(w_low));
            sum = vmlal_s16(sum, vget_high_s16(c_low), vget_high_s16(w_low));
            sum = vmlal_s16(sum, vget_low_s16(c_high), vget_low_s16(w_high));
            sum = vmlal_s16(sum, vget_high_s16(c_high), vget_high_s16(w_high));
        }

        int32x2_t sum_high = vget_high_s32(sum);
        int32x2_t sum_low  = vget_low_s32(sum);
        int32x2_t sum_pair = vadd_s32(sum_high, sum_low);
        int32_t score = vget_lane_s32(vpadd_s32(sum_pair, sum_pair), 0);

        if (coarse_heap.size() < p) {
            coarse_heap.push({(float)score, i});
        } else if (score > (int32_t)coarse_heap.top().first) {
            coarse_heap.pop();
            coarse_heap.push({(float)score, i});
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(coarse_heap.size());
    while (!coarse_heap.empty()) {
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    std::priority_queue<std::pair<float, uint32_t>> final_queue;
    for (uint32_t cand : candidates) {
        const float* base_vec = base + cand * vecdim;
        float dot = ip_96_neon(base_vec, query);
        float dis = 1.0f - dot;

        if (final_queue.size() < k) {
            final_queue.push({dis, cand});
        } else if (dis < final_queue.top().first) {
            final_queue.push({dis, cand});
            final_queue.pop();
        }
    }
    return final_queue;
}