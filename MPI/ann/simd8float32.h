#pragma once
#include <arm_neon.h>
#include <cassert>

struct simd8float32 {
    float32x4x2_t data;  // data.val[0] 低128位, data.val[1] 高128位

    simd8float32() = default;

    // 从连续内存加载8个float
    explicit simd8float32(const float* x) {
        data.val[0] = vld1q_f32(x);
        data.val[1] = vld1q_f32(x + 4);
    }

    // 初始化为标量 v
    explicit simd8float32(float v) {
        data.val[0] = vmovq_n_f32(v);
        data.val[1] = vmovq_n_f32(v);
    }

    // 加法
    simd8float32 operator+(const simd8float32& other) const {
        simd8float32 res;
        res.data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        res.data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return res;
    }

    // 复合加
    simd8float32& operator+=(const simd8float32& other) {
        data.val[0] = vaddq_f32(data.val[0], other.data.val[0]);
        data.val[1] = vaddq_f32(data.val[1], other.data.val[1]);
        return *this;
    }

    // 乘法
    simd8float32 operator*(const simd8float32& other) const {
        simd8float32 res;
        res.data.val[0] = vmulq_f32(data.val[0], other.data.val[0]);
        res.data.val[1] = vmulq_f32(data.val[1], other.data.val[1]);
        return res;
    }

    // 存储到内存
    void store(float* dst) const {
        vst1q_f32(dst, data.val[0]);
        vst1q_f32(dst + 4, data.val[1]);
    }

    // 将8个元素求和为一个float
    float sum() const {
        // 分别对高低各4个求和，再相加
        float32x2_t lo_sum = vadd_f32(vget_high_f32(data.val[0]), vget_low_f32(data.val[0]));
        float32x2_t hi_sum = vadd_f32(vget_high_f32(data.val[1]), vget_low_f32(data.val[1]));
        float32x2_t total4 = vadd_f32(
            vpadd_f32(lo_sum, lo_sum),
            vpadd_f32(hi_sum, hi_sum)
        );
        return vget_lane_f32(vpadd_f32(total4, total4), 0);
    }
};