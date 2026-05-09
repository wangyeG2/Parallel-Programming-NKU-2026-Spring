#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
#include <queue>
#include <utility>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
#include <omp.h>
#include <arm_neon.h>
#include "simd8float32.h"  

struct PQIndex {
    size_t M;          
    size_t Ks;          
    size_t subdim;      
    size_t total_dim;   

    // AoS
    std::vector<float> codebook;
    // SoA
    std::vector<float> codebook_soa;
    std::vector<uint8_t> codes;

    PQIndex() = default;
};
inline float simd_dot_product(const float* x, const float* y, size_t dim) {
    assert(dim % 8 == 0);
    simd8float32 sum(0.0f);
    for (size_t i = 0; i < dim; i += 8) {
        simd8float32 sx(x + i), sy(y + i);
        sum += sx * sy;
    }
    return sum.sum();
}
inline void simd_dot_8centroids_soa(
    const float* subvec,
    const float* centroids_soa,
    size_t subdim,
    size_t Ks,
    size_t c_start,
    float* dots_out)
{
    assert(c_start + 7 < Ks);
    float32x4_t sum_low  = vdupq_n_f32(0.0f);   
    float32x4_t sum_high = vdupq_n_f32(0.0f);   

    for (size_t d = 0; d < subdim; ++d) {
        float sv = subvec[d];
        float32x4_t sv4 = vdupq_n_f32(sv);
        const float* cdim = centroids_soa + d * Ks + c_start;
        float32x4_t c_low  = vld1q_f32(cdim);       
        float32x4_t c_high = vld1q_f32(cdim + 4);   
        sum_low  = vmlaq_f32(sum_low,  sv4, c_low);
        sum_high = vmlaq_f32(sum_high, sv4, c_high);
    }
    vst1q_f32(dots_out,     sum_low);
    vst1q_f32(dots_out + 4, sum_high);
}

inline void simd_dot_8centroids_4subvecs_soa(
    const float* subvec0,
    const float* subvec1,
    const float* subvec2,
    const float* subvec3,
    const float* centroids_soa,
    size_t subdim,
    size_t Ks,
    size_t c_start,
    float* dots_out)  
{
    assert(c_start + 7 < Ks);
    float32x4_t sum0_low  = vdupq_n_f32(0.0f), sum0_high = vdupq_n_f32(0.0f);
    float32x4_t sum1_low  = vdupq_n_f32(0.0f), sum1_high = vdupq_n_f32(0.0f);
    float32x4_t sum2_low  = vdupq_n_f32(0.0f), sum2_high = vdupq_n_f32(0.0f);
    float32x4_t sum3_low  = vdupq_n_f32(0.0f), sum3_high = vdupq_n_f32(0.0f);

    for (size_t d = 0; d < subdim; ++d) {
        const float* cdim = centroids_soa + d * Ks + c_start;
        float32x4_t c_low  = vld1q_f32(cdim);
        float32x4_t c_high = vld1q_f32(cdim + 4);

        float32x4_t sv0 = vdupq_n_f32(subvec0[d]);
        float32x4_t sv1 = vdupq_n_f32(subvec1[d]);
        float32x4_t sv2 = vdupq_n_f32(subvec2[d]);
        float32x4_t sv3 = vdupq_n_f32(subvec3[d]);

        sum0_low  = vmlaq_f32(sum0_low,  sv0, c_low);
        sum0_high = vmlaq_f32(sum0_high, sv0, c_high);
        sum1_low  = vmlaq_f32(sum1_low,  sv1, c_low);
        sum1_high = vmlaq_f32(sum1_high, sv1, c_high);
        sum2_low  = vmlaq_f32(sum2_low,  sv2, c_low);
        sum2_high = vmlaq_f32(sum2_high, sv2, c_high);
        sum3_low  = vmlaq_f32(sum3_low,  sv3, c_low);
        sum3_high = vmlaq_f32(sum3_high, sv3, c_high);
    }

    vst1q_f32(dots_out + 0,  sum0_low);  vst1q_f32(dots_out + 4,  sum0_high);
    vst1q_f32(dots_out + 8,  sum1_low);  vst1q_f32(dots_out + 12, sum1_high);
    vst1q_f32(dots_out + 16, sum2_low);  vst1q_f32(dots_out + 20, sum2_high);
    vst1q_f32(dots_out + 24, sum3_low);  vst1q_f32(dots_out + 28, sum3_high);
}

void build_pq_index(const float* base, size_t base_number, size_t vecdim,
                    PQIndex& index, size_t M = 4, size_t Ks = 256) {
    index.M = M;
    index.Ks = Ks;
    index.subdim = vecdim / M;
    index.total_dim = vecdim;
    assert(vecdim % M == 0 && index.subdim % 8 == 0);

    index.codebook.resize(M * Ks * index.subdim);
    index.codes.resize(base_number * M);
    index.codebook_soa.resize(M * index.subdim * Ks);

    #pragma omp parallel for
    for (size_t m = 0; m < M; ++m) {
        std::vector<float> subspace(base_number * index.subdim);
        for (size_t i = 0; i < base_number; ++i) {
            std::memcpy(subspace.data() + i * index.subdim,
                        base + i * vecdim + m * index.subdim,
                        index.subdim * sizeof(float));
        }

        std::vector<float> centroids_aos(Ks * index.subdim);
        std::mt19937 rng(42 + m);
        std::uniform_int_distribution<size_t> dist(0, base_number - 1);
        for (size_t c = 0; c < Ks; ++c) {
            size_t idx = dist(rng);
            std::memcpy(centroids_aos.data() + c * index.subdim,
                        subspace.data() + idx * index.subdim,
                        index.subdim * sizeof(float));
        }

        std::vector<float> centroids_soa(index.subdim * Ks);
        for (size_t c = 0; c < Ks; ++c) {
            for (size_t d = 0; d < index.subdim; ++d) {
                centroids_soa[d * Ks + c] = centroids_aos[c * index.subdim + d];
            }
        }

        const size_t BLOCK_SIZE = 256;  
        std::vector<size_t> assign(base_number);
        std::vector<size_t> cnt(Ks, 0);

        for (int iter = 0; iter < 15; ++iter) {
            #pragma omp parallel for
            for (size_t block_start = 0; block_start < base_number; block_start += BLOCK_SIZE) {
                size_t block_end = std::min(block_start + BLOCK_SIZE, base_number);
                for (size_t i = block_start; i < block_end; i += 4) {
                    size_t count = (block_end - i) < 4 ? (block_end - i) : 4;
                    const float* sv[4];
                    float dots[4][8]; 
                    for (size_t j = 0; j < count; ++j)
                        sv[j] = subspace.data() + (i + j) * index.subdim;

                    float best_dot[4] = {-1e30f, -1e30f, -1e30f, -1e30f};
                    size_t best_c[4] = {0,0,0,0};

                    for (size_t c = 0; c < Ks; c += 8) {
                        if (count == 4) {
                            simd_dot_8centroids_4subvecs_soa(sv[0], sv[1], sv[2], sv[3],
                                                            centroids_soa.data(),
                                                            index.subdim, Ks, c,
                                                            &dots[0][0]);
                        } else {
                            for (size_t j = 0; j < count; ++j) {
                                simd_dot_8centroids_soa(sv[j], centroids_soa.data(),
                                                        index.subdim, Ks, c,
                                                        dots[j]);
                            }
                        }
                        for (size_t j = 0; j < count; ++j) {
                            for (int t = 0; t < 8; ++t) {
                                if (dots[j][t] > best_dot[j]) {
                                    best_dot[j] = dots[j][t];
                                    best_c[j] = c + t;
                                }
                            }
                        }
                    }
                    for (size_t j = 0; j < count; ++j)
                        assign[i + j] = best_c[j];
                }
            }

            std::fill(cnt.begin(), cnt.end(), 0);
            std::vector<float> new_soa(index.subdim * Ks, 0.0f);
            #pragma omp parallel for
            for (size_t block_start = 0; block_start < base_number; block_start += BLOCK_SIZE) {
                size_t block_end = std::min(block_start + BLOCK_SIZE, base_number);
                std::vector<float> local_new(index.subdim * Ks, 0.0f);
                std::vector<size_t> local_cnt(Ks, 0);

                for (size_t i = block_start; i < block_end; ++i) {
                    size_t c = assign[i];
                    local_cnt[c]++;
                    const float* src = subspace.data() + i * index.subdim;
                    for (size_t d = 0; d < index.subdim; ++d) {
                        local_new[d * Ks + c] += src[d];
                    }
                }
                #pragma omp critical
                {
                    for (size_t c = 0; c < Ks; ++c) {
                        cnt[c] += local_cnt[c];
                        for (size_t d = 0; d < index.subdim; ++d) {
                            new_soa[d * Ks + c] += local_new[d * Ks + c];
                        }
                    }
                }
            }
            for (size_t c = 0; c < Ks; ++c) {
                if (cnt[c] > 0) {
                    float inv = 1.0f / cnt[c];
                    for (size_t d = 0; d < index.subdim; ++d) {
                        centroids_soa[d * Ks + c] = new_soa[d * Ks + c] * inv;
                    }
                }
            }
        }
        float* codebook_aos = index.codebook.data() + m * Ks * index.subdim;
        float* codebook_soa_m = index.codebook_soa.data() + m * index.subdim * Ks;
        for (size_t c = 0; c < Ks; ++c) {
            for (size_t d = 0; d < index.subdim; ++d) {
                float val = centroids_soa[d * Ks + c];
                codebook_aos[c * index.subdim + d] = val;
                codebook_soa_m[d * Ks + c] = val;
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < base_number; ++i) {
            const float* subvec = subspace.data() + i * index.subdim;
            float best_dot = -1e30f;
            uint8_t best_c = 0;
            for (size_t c = 0; c < Ks; c += 8) {
                float dots[8];
                simd_dot_8centroids_soa(subvec, codebook_soa_m,
                                        index.subdim, Ks, c, dots);
                for (int t = 0; t < 8; ++t) {
                    if (dots[t] > best_dot) {
                        best_dot = dots[t];
                        best_c = static_cast<uint8_t>(c + t);
                    }
                }
            }
            index.codes[i * M + m] = best_c;
        }
    }
}

void compute_lut_subspace(const float* query_sub,
                          const float* codebook_soa,  
                          size_t Ks, size_t subdim,
                          float* lut) {
    #pragma omp parallel for
    for (size_t c = 0; c < Ks; c += 8) {
        simd_dot_8centroids_soa(query_sub, codebook_soa,
                                subdim, Ks, c, lut + c);
    }
}
std::vector<uint32_t> pq_coarse_search(const PQIndex& index,
                                       const std::vector<std::vector<float>>& LUTs,
                                       size_t base_number, size_t p) {
    struct Candidate {
        float dist;
        uint32_t id;
        bool operator<(const Candidate& o) const { return dist < o.dist; }
    };
    std::priority_queue<Candidate> heap;
    const size_t M = index.M;
    const uint8_t* codes = index.codes.data();
    const size_t BLOCK_SIZE = 256;

    #pragma omp parallel
    {
        std::priority_queue<Candidate> local_heap;
        #pragma omp for nowait
        for (size_t block_start = 0; block_start < base_number; block_start += BLOCK_SIZE) {
            size_t block_end = std::min(block_start + BLOCK_SIZE, base_number);
            
            for (size_t i = block_start; i < block_end; i += 4) {
                uint8x16_t codes_vec = vld1q_u8(codes + i * M);
                uint32_t raw[4];
                vst1q_u8(reinterpret_cast<uint8_t*>(raw), codes_vec);
                uint8_t idx[4][4];
                for (int j = 0; j < 4; ++j) {
                    uint32_t val = raw[j];
                    idx[j][0] = val & 0xFF;
                    idx[j][1] = (val >> 8) & 0xFF;
                    idx[j][2] = (val >> 16) & 0xFF;
                    idx[j][3] = (val >> 24) & 0xFF;
                }

                float32x4_t dots_vec = vdupq_n_f32(0.0f);
                for (size_t m = 0; m < M; ++m) {
                    const float* lut = LUTs[m].data();
                    float lut_vals[4];
                    if (m + 1 < M) {
                        const float* next_lut = LUTs[m+1].data();
                        __builtin_prefetch(next_lut + idx[0][m+1], 0, 3);
                        __builtin_prefetch(next_lut + idx[1][m+1], 0, 3);
                        __builtin_prefetch(next_lut + idx[2][m+1], 0, 3);
                        __builtin_prefetch(next_lut + idx[3][m+1], 0, 3);
                    }
                    lut_vals[0] = lut[idx[0][m]];
                    lut_vals[1] = lut[idx[1][m]];
                    lut_vals[2] = lut[idx[2][m]];
                    lut_vals[3] = lut[idx[3][m]];
                    float32x4_t lut_vec = vld1q_f32(lut_vals);
                    dots_vec = vaddq_f32(dots_vec, lut_vec);
                }

                float dots[4];
                vst1q_f32(dots, dots_vec);
                for (int j = 0; j < 4 && (i + j) < base_number; ++j) {
                    float dist = 1.0f - dots[j];
                    if (local_heap.size() < p) {
                        local_heap.push({dist, static_cast<uint32_t>(i + j)});
                    } else if (dist < local_heap.top().dist) {
                        local_heap.pop();
                        local_heap.push({dist, static_cast<uint32_t>(i + j)});
                    }
                }
            }
        }

        #pragma omp critical
        {
            while (!local_heap.empty()) {
                auto c = local_heap.top();
                local_heap.pop();
                if (heap.size() < p) {
                    heap.push(c);
                } else if (c.dist < heap.top().dist) {
                    heap.pop();
                    heap.push(c);
                }
            }
        }
    }

    std::vector<uint32_t> candidates;
    candidates.reserve(heap.size());
    while (!heap.empty()) {
        candidates.push_back(heap.top().id);
        heap.pop();
    }
    std::reverse(candidates.begin(), candidates.end());
    return candidates;
}

std::priority_queue<std::pair<float, uint32_t>>
simd_rerank(const float* base, const float* query,
            const std::vector<uint32_t>& candidates,
            size_t vecdim, size_t k) {
    assert(vecdim == 96);
    std::priority_queue<std::pair<float, uint32_t>> q;

    const float32x4_t q0  = vld1q_f32(query);
    const float32x4_t q1  = vld1q_f32(query + 4);
    const float32x4_t q2  = vld1q_f32(query + 8);
    const float32x4_t q3  = vld1q_f32(query + 12);
    const float32x4_t q4  = vld1q_f32(query + 16);
    const float32x4_t q5  = vld1q_f32(query + 20);
    const float32x4_t q6  = vld1q_f32(query + 24);
    const float32x4_t q7  = vld1q_f32(query + 28);
    const float32x4_t q8  = vld1q_f32(query + 32);
    const float32x4_t q9  = vld1q_f32(query + 36);
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

    for (size_t idx : candidates) {
        const float* cur = base + idx * vecdim;
        float32x4_t prod0  = vmulq_f32(vld1q_f32(cur),        q0);
        float32x4_t prod1  = vmulq_f32(vld1q_f32(cur + 4),    q1);
        float32x4_t prod2  = vmulq_f32(vld1q_f32(cur + 8),    q2);
        float32x4_t prod3  = vmulq_f32(vld1q_f32(cur + 12),   q3);
        float32x4_t prod4  = vmulq_f32(vld1q_f32(cur + 16),   q4);
        float32x4_t prod5  = vmulq_f32(vld1q_f32(cur + 20),   q5);
        float32x4_t prod6  = vmulq_f32(vld1q_f32(cur + 24),   q6);
        float32x4_t prod7  = vmulq_f32(vld1q_f32(cur + 28),   q7);
        float32x4_t prod8  = vmulq_f32(vld1q_f32(cur + 32),   q8);
        float32x4_t prod9  = vmulq_f32(vld1q_f32(cur + 36),   q9);
        float32x4_t prod10 = vmulq_f32(vld1q_f32(cur + 40),   q10);
        float32x4_t prod11 = vmulq_f32(vld1q_f32(cur + 44),   q11);
        float32x4_t prod12 = vmulq_f32(vld1q_f32(cur + 48),   q12);
        float32x4_t prod13 = vmulq_f32(vld1q_f32(cur + 52),   q13);
        float32x4_t prod14 = vmulq_f32(vld1q_f32(cur + 56),   q14);
        float32x4_t prod15 = vmulq_f32(vld1q_f32(cur + 60),   q15);
        float32x4_t prod16 = vmulq_f32(vld1q_f32(cur + 64),   q16);
        float32x4_t prod17 = vmulq_f32(vld1q_f32(cur + 68),   q17);
        float32x4_t prod18 = vmulq_f32(vld1q_f32(cur + 72),   q18);
        float32x4_t prod19 = vmulq_f32(vld1q_f32(cur + 76),   q19);
        float32x4_t prod20 = vmulq_f32(vld1q_f32(cur + 80),   q20);
        float32x4_t prod21 = vmulq_f32(vld1q_f32(cur + 84),   q21);
        float32x4_t prod22 = vmulq_f32(vld1q_f32(cur + 88),   q22);
        float32x4_t prod23 = vmulq_f32(vld1q_f32(cur + 92),   q23);

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
        float32x2_t sum_low = vadd_f32(vget_high_f32(sum_total), vget_low_f32(sum_total));
        float dot = vget_lane_f32(vpadd_f32(sum_low, sum_low), 0);

        float dis = 1.0f - dot;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(idx)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(idx)});
            q.pop();
        }
    }
    return q;
}

std::priority_queue<std::pair<float, uint32_t>>
pq_search(const PQIndex& index, const float* base, const float* query,
          size_t base_number, size_t vecdim, size_t k, size_t p) {
    std::vector<std::vector<float>> LUTs(index.M,
                                         std::vector<float>(index.Ks));
    for (size_t m = 0; m < index.M; ++m) {
        const float* query_sub = query + m * index.subdim;
        const float* codebook_soa = index.codebook_soa.data() + m * index.subdim * index.Ks;
        compute_lut_subspace(query_sub, codebook_soa,
                             index.Ks, index.subdim, LUTs[m].data());
    }

    std::vector<uint32_t> candidates = pq_coarse_search(index, LUTs, base_number, p);

    return simd_rerank(base, query, candidates, vecdim, k);
}