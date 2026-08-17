#include "quantize.h"
#include <cmath>
#include <algorithm>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFN_HAVE_NEON 1
#else
#define FFN_HAVE_NEON 0
#endif

namespace ffn {

static inline int8_t quant_one(float v, float inv_scale) {
    int q = static_cast<int>(std::lround(v * inv_scale));
    q = std::max(-127, std::min(127, q)); // symmetric: keep -128 out of range
    return static_cast<int8_t>(q);
}

void quantize_weights_per_channel(const float* W_T, int8_t* Wq_T,
                                   float* scale_out, int N, int K) {
    for (int n = 0; n < N; ++n) {
        const float* row = W_T + static_cast<size_t>(n) * K;
        float max_abs = 1e-12f;
        for (int k = 0; k < K; ++k) max_abs = std::max(max_abs, std::abs(row[k]));
        const float scale = max_abs / 127.0f;
        scale_out[n] = scale;
        const float inv_scale = 1.0f / scale;
        int8_t* qrow = Wq_T + static_cast<size_t>(n) * K;
        for (int k = 0; k < K; ++k) qrow[k] = quant_one(row[k], inv_scale);
    }
}

float quantize_activations_per_tensor(const float* X, int8_t* Xq, int M, int K) {
    const size_t n = static_cast<size_t>(M) * K;
    float max_abs = 1e-12f;
    for (size_t i = 0; i < n; ++i) max_abs = std::max(max_abs, std::abs(X[i]));
    const float scale = max_abs / 127.0f;
    const float inv_scale = 1.0f / scale;
    for (size_t i = 0; i < n; ++i) Xq[i] = quant_one(X[i], inv_scale);
    return scale;
}

#if FFN_HAVE_NEON

static inline int32_t dot_int8_row(const int8_t* a, const int8_t* b, int K) {
    int32_t sum = 0;
    int k = 0;

#if defined(__ARM_FEATURE_DOTPROD)
    // Fast path: vdotq_s32 does 4 independent 4-way int8 dot products per
    // call (16 int8 MACs/instruction), accumulating straight into int32 —
    // this is the instruction Cloud AI 100 / QNN-class kernels lean on.
    int32x4_t acc = vdupq_n_s32(0);
    for (; k + 16 <= K; k += 16) {
        int8x16_t va = vld1q_s8(a + k);
        int8x16_t vb = vld1q_s8(b + k);
        acc = vdotq_s32(acc, va, vb);
    }
    sum = vaddvq_s32(acc);
#else
    // Portable fallback for ARM targets without the dot-product extension:
    // widen 8 int8 lanes to int16 (vmull_s8 — safe, 127*127 fits int16),
    // then widen-accumulate into int32 (vaddw_s16) before the int16 partial
    // sums themselves could overflow.
    int32x4_t acc_lo = vdupq_n_s32(0);
    int32x4_t acc_hi = vdupq_n_s32(0);
    for (; k + 8 <= K; k += 8) {
        int8x8_t va = vld1_s8(a + k);
        int8x8_t vb = vld1_s8(b + k);
        int16x8_t prod = vmull_s8(va, vb);
        acc_lo = vaddw_s16(acc_lo, vget_low_s16(prod));
        acc_hi = vaddw_s16(acc_hi, vget_high_s16(prod));
    }
    sum = vaddvq_s32(vaddq_s32(acc_lo, acc_hi));
#endif

    for (; k < K; ++k) sum += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);
    return sum;
}

#else

static inline int32_t dot_int8_row(const int8_t* a, const int8_t* b, int K) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) sum += static_cast<int32_t>(a[k]) * static_cast<int32_t>(b[k]);
    return sum;
}

#endif

void gemm_int8(const int8_t* Xq, const int8_t* Wq_T, float* C,
               int M, int N, int K, float act_scale, const float* w_scale) {
    for (int m = 0; m < M; ++m) {
        const int8_t* a_row = Xq + static_cast<size_t>(m) * K;
        for (int n = 0; n < N; ++n) {
            const int8_t* b_row = Wq_T + static_cast<size_t>(n) * K;
            const int32_t acc = dot_int8_row(a_row, b_row, K);
            // Requantize: int32 accumulator -> float via per-channel weight
            // scale * per-tensor activation scale.
            C[static_cast<size_t>(m) * N + n] =
                static_cast<float>(acc) * act_scale * w_scale[n];
        }
    }
}

} // namespace ffn
