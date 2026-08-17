#pragma once
// silu.h — SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
//
// NEON has no hardware exp instruction, so this needs a vectorized
// approximation. We use the standard range-reduction + polynomial approach
// (the same technique libm and most fast-math SIMD libraries use):
//   exp(x) = 2^n * exp(r),  where x = n*ln(2) + r,  |r| <= ln(2)/2
// n is recovered as an integer, exp(r) is approximated by a degree-5
// polynomial (accurate to within ~1e-6 relative error over the reduced
// range), and 2^n is reconstructed by directly manipulating the float's
// IEEE-754 exponent bits — no branches, fully vectorizable.
//
// This is the same order of accuracy production kernels (XNNPACK, Eigen)
// target, and is what "<0.5% numerical error vs FP32" needs to actually be
// true rather than asserted — plug scalar_silu_ref and neon-batch results
// into ffn::compare() to verify it for your specific weight/activation
// ranges.

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define FFN_SILU_HAVE_NEON 1
#else
#define FFN_SILU_HAVE_NEON 0
#endif

namespace ffn {

// Scalar reference (uses std::exp) — ground truth for correctness checks.
float silu_scalar_ref(float x);

// Vectorized SiLU over a contiguous buffer, in place semantics via separate
// in/out pointers (may alias). NEON path processes 4 floats/iteration with
// scalar cleanup for the remainder; falls back to silu_scalar_ref on non-ARM.
void silu_inplace(const float* in, float* out, int n);

#if FFN_SILU_HAVE_NEON
// Register-level SiLU on exactly 4 lanes — the primitive fused_ffn.cpp uses
// to apply SiLU without ever spilling the gate projection to memory.
float32x4_t silu_neon4(float32x4_t x);
#endif

} // namespace ffn
