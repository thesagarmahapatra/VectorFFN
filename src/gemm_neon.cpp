#include "gemm_neon.h"
#include <vector>
#include <algorithm>
#include <cstring>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFN_HAVE_NEON 1
#else
#define FFN_HAVE_NEON 0
#endif

namespace ffn {

#if FFN_HAVE_NEON

// 8-wide microkernel: one A row against 8 B_T rows simultaneously.
// Keeps 8 vector accumulators in registers (v0..v7), loading 1 A vector and 8 B vectors.
// Delivers 8 FMAs per A-load, maximizing arithmetic intensity on Cortex-X3 / A715 / M1 Firestorm.
static inline void microkernel_8x1(const float* a_row,
                                    const float* b0, const float* b1,
                                    const float* b2, const float* b3,
                                    const float* b4, const float* b5,
                                    const float* b6, const float* b7,
                                    int kb, float* acc /*[8]*/) {
    float32x4_t v_acc0 = vdupq_n_f32(0.0f);
    float32x4_t v_acc1 = vdupq_n_f32(0.0f);
    float32x4_t v_acc2 = vdupq_n_f32(0.0f);
    float32x4_t v_acc3 = vdupq_n_f32(0.0f);
    float32x4_t v_acc4 = vdupq_n_f32(0.0f);
    float32x4_t v_acc5 = vdupq_n_f32(0.0f);
    float32x4_t v_acc6 = vdupq_n_f32(0.0f);
    float32x4_t v_acc7 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k + 4 <= kb; k += 4) {
        float32x4_t va = vld1q_f32(a_row + k);
        v_acc0 = vfmaq_f32(v_acc0, va, vld1q_f32(b0 + k));
        v_acc1 = vfmaq_f32(v_acc1, va, vld1q_f32(b1 + k));
        v_acc2 = vfmaq_f32(v_acc2, va, vld1q_f32(b2 + k));
        v_acc3 = vfmaq_f32(v_acc3, va, vld1q_f32(b3 + k));
        v_acc4 = vfmaq_f32(v_acc4, va, vld1q_f32(b4 + k));
        v_acc5 = vfmaq_f32(v_acc5, va, vld1q_f32(b5 + k));
        v_acc6 = vfmaq_f32(v_acc6, va, vld1q_f32(b6 + k));
        v_acc7 = vfmaq_f32(v_acc7, va, vld1q_f32(b7 + k));
    }

    acc[0] += vaddvq_f32(v_acc0);
    acc[1] += vaddvq_f32(v_acc1);
    acc[2] += vaddvq_f32(v_acc2);
    acc[3] += vaddvq_f32(v_acc3);
    acc[4] += vaddvq_f32(v_acc4);
    acc[5] += vaddvq_f32(v_acc5);
    acc[6] += vaddvq_f32(v_acc6);
    acc[7] += vaddvq_f32(v_acc7);

    for (; k < kb; ++k) {
        float av = a_row[k];
        acc[0] += av * b0[k];
        acc[1] += av * b1[k];
        acc[2] += av * b2[k];
        acc[3] += av * b3[k];
        acc[4] += av * b4[k];
        acc[5] += av * b5[k];
        acc[6] += av * b6[k];
        acc[7] += av * b7[k];
    }
}

// 4-wide microkernel: one A row against 4 B_T rows simultaneously.
static inline void microkernel_4x1(const float* a_row, const float* b_row0,
                                    const float* b_row1, const float* b_row2,
                                    const float* b_row3, int kb,
                                    float* acc /*[4]*/) {
    float32x4_t v_acc0 = vdupq_n_f32(0.0f);
    float32x4_t v_acc1 = vdupq_n_f32(0.0f);
    float32x4_t v_acc2 = vdupq_n_f32(0.0f);
    float32x4_t v_acc3 = vdupq_n_f32(0.0f);

    int k = 0;
    for (; k + 4 <= kb; k += 4) {
        float32x4_t va = vld1q_f32(a_row + k);
        v_acc0 = vfmaq_f32(v_acc0, va, vld1q_f32(b_row0 + k));
        v_acc1 = vfmaq_f32(v_acc1, va, vld1q_f32(b_row1 + k));
        v_acc2 = vfmaq_f32(v_acc2, va, vld1q_f32(b_row2 + k));
        v_acc3 = vfmaq_f32(v_acc3, va, vld1q_f32(b_row3 + k));
    }

    acc[0] += vaddvq_f32(v_acc0);
    acc[1] += vaddvq_f32(v_acc1);
    acc[2] += vaddvq_f32(v_acc2);
    acc[3] += vaddvq_f32(v_acc3);

    for (; k < kb; ++k) {
        float av = a_row[k];
        acc[0] += av * b_row0[k];
        acc[1] += av * b_row1[k];
        acc[2] += av * b_row2[k];
        acc[3] += av * b_row3[k];
    }
}

static inline float dot_scalar(const float* a, const float* b, int kb) {
    float32x4_t v_acc = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 4 <= kb; k += 4) {
        v_acc = vfmaq_f32(v_acc, vld1q_f32(a + k), vld1q_f32(b + k));
    }
    float s = vaddvq_f32(v_acc);
    for (; k < kb; ++k) s += a[k] * b[k];
    return s;
}

void gemm_neon(const float* A, const float* B_T, float* C,
               int M, int N, int K, int MT, int NT, int KT) {
    std::vector<float> acc_tile(static_cast<size_t>(MT) * NT);

    for (int i0 = 0; i0 < M; i0 += MT) {
        const int ib = std::min(MT, M - i0);
        for (int j0 = 0; j0 < N; j0 += NT) {
            const int jb = std::min(NT, N - j0);
            std::fill(acc_tile.begin(), acc_tile.begin() + static_cast<size_t>(ib) * jb, 0.0f);

            for (int k0 = 0; k0 < K; k0 += KT) {
                const int kb = std::min(KT, K - k0);

                for (int m = 0; m < ib; ++m) {
                    const float* a_row = A + static_cast<size_t>(i0 + m) * K + k0;
                    float* acc_row = &acc_tile[static_cast<size_t>(m) * jb];

                    int n = 0;
                    // 8-wide unrolled vector microkernel
                    for (; n + 8 <= jb; n += 8) {
                        const float* b0 = B_T + static_cast<size_t>(j0 + n + 0) * K + k0;
                        const float* b1 = B_T + static_cast<size_t>(j0 + n + 1) * K + k0;
                        const float* b2 = B_T + static_cast<size_t>(j0 + n + 2) * K + k0;
                        const float* b3 = B_T + static_cast<size_t>(j0 + n + 3) * K + k0;
                        const float* b4 = B_T + static_cast<size_t>(j0 + n + 4) * K + k0;
                        const float* b5 = B_T + static_cast<size_t>(j0 + n + 5) * K + k0;
                        const float* b6 = B_T + static_cast<size_t>(j0 + n + 6) * K + k0;
                        const float* b7 = B_T + static_cast<size_t>(j0 + n + 7) * K + k0;
                        microkernel_8x1(a_row, b0, b1, b2, b3, b4, b5, b6, b7, kb, &acc_row[n]);
                    }
                    // 4-wide vector microkernel
                    for (; n + 4 <= jb; n += 4) {
                        const float* b0 = B_T + static_cast<size_t>(j0 + n + 0) * K + k0;
                        const float* b1 = B_T + static_cast<size_t>(j0 + n + 1) * K + k0;
                        const float* b2 = B_T + static_cast<size_t>(j0 + n + 2) * K + k0;
                        const float* b3 = B_T + static_cast<size_t>(j0 + n + 3) * K + k0;
                        microkernel_4x1(a_row, b0, b1, b2, b3, kb, &acc_row[n]);
                    }
                    // Scalar cleanup (NEON along K)
                    for (; n < jb; ++n) {
                        const float* b_row = B_T + static_cast<size_t>(j0 + n) * K + k0;
                        acc_row[n] += dot_scalar(a_row, b_row, kb);
                    }
                }
            }

            for (int m = 0; m < ib; ++m) {
                float* c_row = C + static_cast<size_t>(i0 + m) * N + j0;
                const float* a_row = &acc_tile[static_cast<size_t>(m) * jb];
                std::memcpy(c_row, a_row, sizeof(float) * jb);
            }
        }
    }
}

void dot4(const float* a, const float* b0, const float* b1,
          const float* b2, const float* b3, int K, float out[4]) {
    out[0] = out[1] = out[2] = out[3] = 0.0f;
    microkernel_4x1(a, b0, b1, b2, b3, K, out);
}

void dot8(const float* a, const float* b0, const float* b1,
          const float* b2, const float* b3, const float* b4,
          const float* b5, const float* b6, const float* b7,
          int K, float out[8]) {
    for (int i = 0; i < 8; ++i) out[i] = 0.0f;
    microkernel_8x1(a, b0, b1, b2, b3, b4, b5, b6, b7, K, out);
}

#else // !FFN_HAVE_NEON — portable fallback so the project still builds on x86 for CI

void gemm_neon(const float* A, const float* B_T, float* C,
               int M, int N, int K, int MT, int NT, int KT) {
    std::vector<float> acc_tile(static_cast<size_t>(MT) * NT);
    for (int i0 = 0; i0 < M; i0 += MT) {
        const int ib = std::min(MT, M - i0);
        for (int j0 = 0; j0 < N; j0 += NT) {
            const int jb = std::min(NT, N - j0);
            std::fill(acc_tile.begin(), acc_tile.begin() + static_cast<size_t>(ib) * jb, 0.0f);
            for (int k0 = 0; k0 < K; k0 += KT) {
                const int kb = std::min(KT, K - k0);
                for (int m = 0; m < ib; ++m) {
                    const float* a_row = A + static_cast<size_t>(i0 + m) * K + k0;
                    for (int n = 0; n < jb; ++n) {
                        const float* b_row = B_T + static_cast<size_t>(j0 + n) * K + k0;
                        float s = 0.0f;
                        for (int k = 0; k < kb; ++k) s += a_row[k] * b_row[k];
                        acc_tile[static_cast<size_t>(m) * jb + n] += s;
                    }
                }
            }
            for (int m = 0; m < ib; ++m) {
                std::memcpy(C + static_cast<size_t>(i0 + m) * N + j0,
                            &acc_tile[static_cast<size_t>(m) * jb], sizeof(float) * jb);
            }
        }
    }
}

#endif

} // namespace ffn
