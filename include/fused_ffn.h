#pragma once
// fused_ffn.h — Levels 3 & 4: OpenMP multithreading + register-level fusion.
//
// FFN(x) = (SiLU(x @ Wg_T^T) ⊙ (x @ Wu_T^T)) @ Wd_T^T
// (weights pre-transposed, see common.h for why: [d_ff,d_model] for
// gate/up, [d_model,d_ff] for down.)
//
// Two implementations of the exact same math, for A/B measurement:
//
//   unfused_ffn_forward: materializes gate[seq,d_ff] and up[seq,d_ff] as
//     separate DRAM-resident buffers, combines them into h[seq,d_ff]
//     (another DRAM buffer), then down-projects. This is what you get if
//     you literally call gemm_neon() three times back to back — correct,
//     but every one of those intermediate tensors round-trips through
//     DRAM. This is the "before" side of the DRAM-traffic-reduction claim.
//
//   fused_ffn_forward: per token, computes 4 gate-channels and 4
//     up-channels at once directly in NEON registers (via gemm_neon::dot4,
//     reused rather than reimplemented), applies SiLU and the Hadamard
//     product on those registers, and ONLY THEN writes the combined result
//     to the (unavoidable — down-proj needs the full row) h[] buffer. The
//     full [seq,d_ff] gate and up matrices are never separately
//     materialized in DRAM at all.
//
// Parallelized over the token (sequence) dimension via OpenMP. Uses
// `schedule(dynamic)` rather than the default static split deliberately:
// on heterogeneous cores (Apple Firestorm/Icestorm, or Snapdragon's
// X3/A7xx/A510 split), a static equal-sized chunk-per-thread assignment
// stalls on whichever thread lands on a slow core with a full-size chunk.
// Dynamic scheduling lets fast cores pull more chunks and lets slow cores
// naturally finish with less work, instead of the whole parallel region
// waiting on the slowest static assignment. Benchmark both schedules on
// your actual hardware — see benchmark_main.cpp's --schedule flag.

#include "common.h"

namespace ffn {

struct FFNWeights {
    const float* Wg_T; // [d_ff, d_model]
    const float* Wu_T; // [d_ff, d_model]
    const float* Wd_T; // [d_model, d_ff]
};

void naive_ffn_forward(const float* X, const FFNWeights& W, float* out,
                        const FFNShape& shape);

void unfused_ffn_forward(const float* X, const FFNWeights& W, float* out,
                          const FFNShape& shape);

// omp_schedule_dynamic: true => schedule(dynamic), false => schedule(static).
// Exposed as a runtime flag specifically so you can benchmark the P/E-core
// scheduling difference discussed above without recompiling.
void fused_ffn_forward(const float* X, const FFNWeights& W, float* out,
                        const FFNShape& shape, bool omp_schedule_dynamic = true);

} // namespace ffn
