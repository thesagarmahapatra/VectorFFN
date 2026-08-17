#pragma once
// quantize.h — Level 5: symmetric INT8 quantization + vdotq_s32 GEMM.
//
// Weights: per-CHANNEL (per output row of B_T) symmetric quantization.
//   scale_w[n] = max(|W_T[n, :]|) / 127
//   Wq_T[n,k]  = clamp(round(W_T[n,k] / scale_w[n]), -127, 127)
// Per-channel (rather than one scale for the whole tensor) matters because
// weight magnitude distributions vary a lot across output channels — a
// single per-tensor scale wastes precision on channels far from the tensor
// max, which is exactly what tanks accuracy in naive INT8 quantization.
//
// Activations: per-TENSOR symmetric quantization (one scale for the whole
// input row/batch). Per-channel activation quantization is possible but
// requires requantizing on every forward pass since activations change
// per input, which is why per-tensor (computed once from a calibration
// pass, or dynamically per batch) is the standard choice for the
// activation side.
//
// GEMM accumulation: int8 x int8 -> int32 via ARM's vdotq_s32 (4-way
// dot-product instruction, ARMv8.2+dotprod — present on M1 since it's
// ARMv8.5-A, and on Snapdragon 8 Gen 2's Cortex-X3/A7xx cores). Gated by
// __ARM_FEATURE_DOTPROD with a portable widening-multiply fallback for
// older ARM targets that lack the dot-product extension.
//
// Overflow check (do this math for YOUR shapes, don't trust this comment):
// worst case per MAC is 127*127 = 16129; accumulating over K=d_model
// (e.g. 4096-14336 for LLaMA/Mistral-scale FFNs) into int32 gives a
// worst-case sum on the order of 16129 * 14336 ~= 2.3e8, which is well
// under INT32_MAX (~2.1e9) — no overflow for realistic FFN K dimensions.
// This stops being true if K gets much larger or if accumulation happens
// in int16 instead of int32 — always re-derive this for your actual shape.

#include <cstdint>

namespace ffn {

// Per-channel weight quantization. W_T: [N,K] row-major (N channels, each a
// contiguous row). Wq_T: same shape, int8 output. scale_out: [N] output.
void quantize_weights_per_channel(const float* W_T, int8_t* Wq_T,
                                   float* scale_out, int N, int K);

// Per-tensor activation quantization. X: [M,K]. Xq: same shape, int8 output.
// Returns the single scale used.
float quantize_activations_per_tensor(const float* X, int8_t* Xq, int M, int K);

// C[m,n] = dequant( sum_k Xq[m,k] * Wq_T[n,k] ), using vdotq_s32 where
// available. act_scale is the single activation scale; w_scale[n] is the
// per-channel weight scale for output channel n.
void gemm_int8(const int8_t* Xq, const int8_t* Wq_T, float* C,
               int M, int N, int K, float act_scale, const float* w_scale);

} // namespace ffn
