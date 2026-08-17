#include "fused_ffn.h"
#include "gemm_neon.h"
#include "silu.h"
#include <vector>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFN_HAVE_NEON 1
#else
#define FFN_HAVE_NEON 0
#endif

namespace ffn {

// ---------------------------------------------------------------------------
// Level 0 Naive scalar baseline: three separate scalar GEMMs + DRAM h.
// ---------------------------------------------------------------------------
void naive_ffn_forward(const float* X, const FFNWeights& W, float* out,
                        const FFNShape& shape) {
    const int M = shape.seq_len, K = shape.d_model, F = shape.d_ff;

    AlignedBuffer<float> gate(static_cast<size_t>(M) * F);
    AlignedBuffer<float> up(static_cast<size_t>(M) * F);
    AlignedBuffer<float> h(static_cast<size_t>(M) * F);

    for (int m = 0; m < M; ++m) {
        for (int f = 0; f < F; ++f) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += X[static_cast<size_t>(m) * K + k] * W.Wg_T[static_cast<size_t>(f) * K + k];
            }
            gate[static_cast<size_t>(m) * F + f] = s;
        }
    }

    for (int m = 0; m < M; ++m) {
        for (int f = 0; f < F; ++f) {
            float s = 0.0f;
            for (int k = 0; k < K; ++k) {
                s += X[static_cast<size_t>(m) * K + k] * W.Wu_T[static_cast<size_t>(f) * K + k];
            }
            up[static_cast<size_t>(m) * F + f] = s;
        }
    }

    for (size_t i = 0; i < static_cast<size_t>(M) * F; ++i) {
        h[i] = silu_scalar_ref(gate[i]) * up[i];
    }

    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            float s = 0.0f;
            for (int f = 0; f < F; ++f) {
                s += h[static_cast<size_t>(m) * F + f] * W.Wd_T[static_cast<size_t>(k) * F + f];
            }
            out[static_cast<size_t>(m) * K + k] = s;
        }
    }
}

// ---------------------------------------------------------------------------
// "Before": three separate GEMMs, each intermediate fully materialized.
// DRAM traffic for the gate/up/combine stage alone (ignoring the down-proj,
// which both versions pay identically):
//   write gate[seq,d_ff]  -> seq*d_ff*4 bytes
//   write up[seq,d_ff]    -> seq*d_ff*4 bytes
//   read  gate + read up + write h[seq,d_ff] (elementwise combine)
//                          -> 3 * seq*d_ff*4 bytes
//   read h (consumed by down-proj GEMM)
//                          -> seq*d_ff*4 bytes
// Total: 6 * seq*d_ff*4 bytes of DRAM traffic just for the intermediate
// tensor. fused_ffn_forward below replaces all of this with a single write
// of h (1x), i.e. an ~83% reduction on this stage — see benchmark_main.cpp
// for the actual measured figure on your hardware (theoretical bandwidth
// savings and observed latency savings are NOT the same number once you
// account for cache effects; report what you measure, not this estimate).
// ---------------------------------------------------------------------------
void unfused_ffn_forward(const float* X, const FFNWeights& W, float* out,
                          const FFNShape& shape) {
    const int M = shape.seq_len, K = shape.d_model, F = shape.d_ff;

    AlignedBuffer<float> gate(static_cast<size_t>(M) * F);
    AlignedBuffer<float> up(static_cast<size_t>(M) * F);
    AlignedBuffer<float> h(static_cast<size_t>(M) * F);

    gemm_neon(X, W.Wg_T, gate.data(), M, F, K);   // materialize gate -> DRAM
    gemm_neon(X, W.Wu_T, up.data(), M, F, K);     // materialize up   -> DRAM

    silu_inplace(gate.data(), gate.data(), M * F); // read+write gate -> DRAM

    for (int i = 0; i < M * F; ++i) {              // read gate,up; write h
        h[i] = gate[i] * up[i];
    }

    gemm_neon(h.data(), W.Wd_T, out, M, K, F);     // read h -> DRAM
}

// ---------------------------------------------------------------------------
// "After": gate/up computed 4 channels at a time directly into NEON
// registers, SiLU + Hadamard applied on those registers, and only the
// final combined value touches memory (into the per-token h[] scratch,
// which is unavoidable — the down-projection GEMM needs the full d_ff
// vector as its input row).
// ---------------------------------------------------------------------------
#if FFN_HAVE_NEON

static void fused_token_row(const float* x_row, const FFNWeights& W,
                             float* h, int d_model, int d_ff) {
    int n = 0;
    // 8-wide fusion
    for (; n + 8 <= d_ff; n += 8) {
        float gate8[8], up8[8];
        dot8(x_row,
             W.Wg_T + static_cast<size_t>(n + 0) * d_model,
             W.Wg_T + static_cast<size_t>(n + 1) * d_model,
             W.Wg_T + static_cast<size_t>(n + 2) * d_model,
             W.Wg_T + static_cast<size_t>(n + 3) * d_model,
             W.Wg_T + static_cast<size_t>(n + 4) * d_model,
             W.Wg_T + static_cast<size_t>(n + 5) * d_model,
             W.Wg_T + static_cast<size_t>(n + 6) * d_model,
             W.Wg_T + static_cast<size_t>(n + 7) * d_model,
             d_model, gate8);
        dot8(x_row,
             W.Wu_T + static_cast<size_t>(n + 0) * d_model,
             W.Wu_T + static_cast<size_t>(n + 1) * d_model,
             W.Wu_T + static_cast<size_t>(n + 2) * d_model,
             W.Wu_T + static_cast<size_t>(n + 3) * d_model,
             W.Wu_T + static_cast<size_t>(n + 4) * d_model,
             W.Wu_T + static_cast<size_t>(n + 5) * d_model,
             W.Wu_T + static_cast<size_t>(n + 6) * d_model,
             W.Wu_T + static_cast<size_t>(n + 7) * d_model,
             d_model, up8);

        float32x4_t g0 = vld1q_f32(gate8);
        float32x4_t g1 = vld1q_f32(gate8 + 4);
        float32x4_t u0 = vld1q_f32(up8);
        float32x4_t u1 = vld1q_f32(up8 + 4);

        vst1q_f32(h + n + 0, vmulq_f32(silu_neon4(g0), u0));
        vst1q_f32(h + n + 4, vmulq_f32(silu_neon4(g1), u1));
    }
    // 4-wide fusion
    for (; n + 4 <= d_ff; n += 4) {
        float gate4[4], up4[4];
        dot4(x_row,
             W.Wg_T + static_cast<size_t>(n + 0) * d_model,
             W.Wg_T + static_cast<size_t>(n + 1) * d_model,
             W.Wg_T + static_cast<size_t>(n + 2) * d_model,
             W.Wg_T + static_cast<size_t>(n + 3) * d_model,
             d_model, gate4);
        dot4(x_row,
             W.Wu_T + static_cast<size_t>(n + 0) * d_model,
             W.Wu_T + static_cast<size_t>(n + 1) * d_model,
             W.Wu_T + static_cast<size_t>(n + 2) * d_model,
             W.Wu_T + static_cast<size_t>(n + 3) * d_model,
             d_model, up4);

        float32x4_t g = vld1q_f32(gate4);
        float32x4_t u = vld1q_f32(up4);
        vst1q_f32(h + n, vmulq_f32(silu_neon4(g), u));
    }
    for (; n < d_ff; ++n) { // remainder, scalar
        float g = 0.0f, u = 0.0f;
        const float* wg = W.Wg_T + static_cast<size_t>(n) * d_model;
        const float* wu = W.Wu_T + static_cast<size_t>(n) * d_model;
        for (int k = 0; k < d_model; ++k) {
            g += x_row[k] * wg[k];
            u += x_row[k] * wu[k];
        }
        h[n] = silu_scalar_ref(g) * u;
    }
}

// Down-projection for a single token row: out_row = h @ Wd_T^T
static void down_project_row(const float* h, const float* Wd_T,
                              float* out_row, int F, int d_model) {
    int k = 0;
    // 8-wide down projection
    for (; k + 8 <= d_model; k += 8) {
        float o8[8];
        dot8(h,
             Wd_T + static_cast<size_t>(k + 0) * F,
             Wd_T + static_cast<size_t>(k + 1) * F,
             Wd_T + static_cast<size_t>(k + 2) * F,
             Wd_T + static_cast<size_t>(k + 3) * F,
             Wd_T + static_cast<size_t>(k + 4) * F,
             Wd_T + static_cast<size_t>(k + 5) * F,
             Wd_T + static_cast<size_t>(k + 6) * F,
             Wd_T + static_cast<size_t>(k + 7) * F,
             F, o8);
        for (int i = 0; i < 8; ++i) out_row[k + i] = o8[i];
    }
    // 4-wide down projection
    for (; k + 4 <= d_model; k += 4) {
        float o4[4];
        dot4(h,
             Wd_T + static_cast<size_t>(k + 0) * F,
             Wd_T + static_cast<size_t>(k + 1) * F,
             Wd_T + static_cast<size_t>(k + 2) * F,
             Wd_T + static_cast<size_t>(k + 3) * F,
             F, o4);
        out_row[k + 0] = o4[0]; out_row[k + 1] = o4[1];
        out_row[k + 2] = o4[2]; out_row[k + 3] = o4[3];
    }
    for (; k < d_model; ++k) {
        const float* wd = Wd_T + static_cast<size_t>(k) * F;
        float s = 0.0f;
        for (int n = 0; n < F; ++n) s += h[n] * wd[n];
        out_row[k] = s;
    }
}

void fused_ffn_forward(const float* X, const FFNWeights& W, float* out,
                        const FFNShape& shape, bool omp_schedule_dynamic) {
    const int M = shape.seq_len, K = shape.d_model, F = shape.d_ff;

#ifdef _OPENMP
    const int nthreads = omp_get_max_threads();
#else
    const int nthreads = 1;
#endif
    // Per-thread scratch, sized once up front — avoids malloc/free inside
    // the hot per-token loop (a fresh heap allocation per token would
    // dominate runtime for small d_ff and defeat the point of fusion).
    std::vector<float> scratch(static_cast<size_t>(nthreads) * F);

#ifdef _OPENMP
    if (omp_schedule_dynamic) {
        #pragma omp parallel for schedule(dynamic)
        for (int m = 0; m < M; ++m) {
            const int tid = omp_get_thread_num();
            float* h = &scratch[static_cast<size_t>(tid) * F];
            fused_token_row(X + static_cast<size_t>(m) * K, W, h, K, F);
            down_project_row(h, W.Wd_T, out + static_cast<size_t>(m) * K, F, K);
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int m = 0; m < M; ++m) {
            const int tid = omp_get_thread_num();
            float* h = &scratch[static_cast<size_t>(tid) * F];
            fused_token_row(X + static_cast<size_t>(m) * K, W, h, K, F);
            down_project_row(h, W.Wd_T, out + static_cast<size_t>(m) * K, F, K);
        }
    }
#else
    (void)omp_schedule_dynamic;
    for (int m = 0; m < M; ++m) {
        float* h = &scratch[0];
        fused_token_row(X + static_cast<size_t>(m) * K, W, h, K, F);
        down_project_row(h, W.Wd_T, out + static_cast<size_t>(m) * K, F, K);
    }
#endif
}

#else // !FFN_HAVE_NEON — scalar portable fallback

void fused_ffn_forward(const float* X, const FFNWeights& W, float* out,
                        const FFNShape& shape, bool /*omp_schedule_dynamic*/) {
    unfused_ffn_forward(X, W, out, shape); // no NEON registers to fuse into
}

#endif

} // namespace ffn
