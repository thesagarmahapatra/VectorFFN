#pragma once
// gemm_neon.h — Level 2: NEON FP32 SIMD on top of the tiled/transposed layout.
//
// Micro-kernel: for each A row, process 4 output columns (N) at once, each
// with its own 128-bit accumulator (v_acc0..v_acc3). That's a deliberate
// register-blocking choice: 4 accumulators + 1 shared A-vector register +
// (up to) 4 B-row vector registers = ~9 live NEON registers per inner
// iteration, well under the 32 available (v0-v31), leaving headroom for the
// compiler's own loop bookkeeping without spilling to stack. Widening this
// to an 8-wide or 6x8 microkernel (BLIS-style) would use more registers for
// more reuse per A-row load, at the cost of more spill risk — worth trying
// as a follow-up and comparing register allocation with `-S`.
//
// Same transposed-B, 3-level-tiled structure as gemm_tiled.h; the
// (MT,NT,KT) cache-blocking wraps this vectorized micro-kernel instead of
// the scalar one.

#include <cstdint>

namespace ffn {

void gemm_neon(const float* A, const float* B_T, float* C,
                int M, int N, int K,
                int MT = 64, int NT = 64, int KT = 256);

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
// Exposed primitive: dot product of `a` against 4 rows (b0..b3) of length K,
// written into out[0..3]. This is the register-blocked microkernel used
// both inside gemm_neon's tiling AND directly by fused_ffn.cpp — the same
// code that fuses Gate/Up projections reuses this rather than reimplementing
// dot products a second time.
void dot4(const float* a, const float* b0, const float* b1,
          const float* b2, const float* b3, int K, float out[4]);

void dot8(const float* a, const float* b0, const float* b1,
          const float* b2, const float* b3, const float* b4,
          const float* b5, const float* b6, const float* b7,
          int K, float out[8]);
#endif

} // namespace ffn
