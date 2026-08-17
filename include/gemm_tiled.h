#pragma once
// gemm_tiled.h — Level 1: cache blocking + pre-transposed weights.
//
// B is now stored TRANSPOSED as B_T [N,K] (this is the one-time offline cost
// every real inference engine pays — weights are transposed once at load
// time, same spirit as folding BatchNorm into Conv). With B_T, computing
// C[m,n] becomes a dot product of two ROW-MAJOR CONTIGUOUS vectors:
//   C[m,n] = sum_k A[m,k] * B_T[n,k]
// Both operands are now unit-stride. That alone fixes most of naive's
// problem. Tiling on top of that is about TEMPORAL reuse, not spatial:
// we block over (M,N,K) so that a tile of B_T rows stays resident in cache
// while we sweep multiple A rows against it, instead of re-fetching B_T
// from DRAM once per (m,n) pair.
//
// Tile sizing math (see README for the derivation): M1 Firestorm has a
// 128KB L1D. A working set of one A-row-slice + one B_T-tile + a float
// accumulator tile needs to fit with headroom for associativity conflicts.
// KT=256 (256 floats = 1KB per row slice) and NT=64 keeps the B_T tile at
// NT*KT*4 = 64KB, comfortably inside L1 with room for A's slice and the
// accumulator. These are exposed as parameters specifically so you can
// re-derive and re-tune them for your actual target cache sizes rather than
// trusting a hardcoded constant.

namespace ffn {

void gemm_tiled(const float* A, const float* B_T, float* C,
                 int M, int N, int K,
                 int MT = 64, int NT = 64, int KT = 256);

} // namespace ffn
