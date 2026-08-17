#pragma once
// gemm_naive.h — Level 0 baseline.
//
// Deliberately the "bad" version: B is stored [K,N] (standard, NOT
// transposed), and the loop order is i-j-k. For fixed (i,j), the inner loop
// over k strides through B with stride N*sizeof(float) — i.e. every access
// touches a different cache line. This is the textbook GEMM everyone writes
// first, and the thing every later kernel in this project fixes.
//
// A: [M,K] row-major.  B: [K,N] row-major.  C: [M,N] row-major (overwritten).

namespace ffn {

void gemm_naive_ijk(const float* A, const float* B, float* C,
                     int M, int N, int K);

} // namespace ffn
