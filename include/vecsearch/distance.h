#pragma once
// SIMD distance kernels. NEON on Apple Silicon / ARM64, AVX2+FMA on x86, scalar otherwise.

#include <cstddef>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define VS_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define VS_AVX2 1
#endif

namespace vs {

inline float l2_sq_scalar(const float* a, const float* b, size_t d) {
    float s = 0.f;
    for (size_t i = 0; i < d; ++i) {
        float t = a[i] - b[i];
        s += t * t;
    }
    return s;
}

inline float dot_scalar(const float* a, const float* b, size_t d) {
    float s = 0.f;
    for (size_t i = 0; i < d; ++i) s += a[i] * b[i];
    return s;
}

inline float l2_sq(const float* a, const float* b, size_t d) {
#if VS_NEON
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
        float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
        acc0 = vfmaq_f32(acc0, d0, d0);
        acc1 = vfmaq_f32(acc1, d1, d1);
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    return s + l2_sq_scalar(a + i, b + i, d - i);
#elif VS_AVX2
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        __m256 t = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_fmadd_ps(t, t, acc);
    }
    alignas(32) float buf[8];
    _mm256_store_ps(buf, acc);
    float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    return s + l2_sq_scalar(a + i, b + i, d - i);
#else
    return l2_sq_scalar(a, b, d);
#endif
}

inline float dot(const float* a, const float* b, size_t d) {
#if VS_NEON
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    size_t i = 0;
    for (; i + 8 <= d; i += 8) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(a + i), vld1q_f32(b + i));
        acc1 = vfmaq_f32(acc1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    }
    float s = vaddvq_f32(vaddq_f32(acc0, acc1));
    return s + dot_scalar(a + i, b + i, d - i);
#elif VS_AVX2
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= d; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
    alignas(32) float buf[8];
    _mm256_store_ps(buf, acc);
    float s = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
    return s + dot_scalar(a + i, b + i, d - i);
#else
    return dot_scalar(a, b, d);
#endif
}

}  // namespace vs
