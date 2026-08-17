#include "gemm_naive.h"

namespace ffn {

void gemm_naive_ijk(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k) {
                // B[k*N + j]: stride-N access. Every iteration is a fresh
                // cache line for any N > 16 (64B line / 4B float). This is
                // the single line that makes this kernel bad.
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

} // namespace ffn
