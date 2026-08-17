#include "gemm_tiled.h"
#include <vector>
#include <algorithm>
#include <cstring>

namespace ffn {

void gemm_tiled(const float* A, const float* B_T, float* C,
                 int M, int N, int K,
                 int MT, int NT, int KT) {
    std::vector<float> acc_tile(static_cast<size_t>(MT) * NT);

    for (int i0 = 0; i0 < M; i0 += MT) {
        const int ib = std::min(MT, M - i0);
        for (int j0 = 0; j0 < N; j0 += NT) {
            const int jb = std::min(NT, N - j0);

            // Zero the local accumulator tile. This lives in L1/registers
            // for the duration of the K sweep below, so C only gets ONE
            // DRAM write per tile instead of one per K-block.
            std::fill(acc_tile.begin(), acc_tile.begin() + static_cast<size_t>(ib) * jb, 0.0f);

            for (int k0 = 0; k0 < K; k0 += KT) {
                const int kb = std::min(KT, K - k0);

                for (int m = 0; m < ib; ++m) {
                    const float* a_row = A + static_cast<size_t>(i0 + m) * K + k0;
                    for (int n = 0; n < jb; ++n) {
                        const float* b_row = B_T + static_cast<size_t>(j0 + n) * K + k0;
                        float s = 0.0f;
                        // Unit-stride dot product on both operands — this is
                        // the line that actually differs from gemm_naive.
                        for (int k = 0; k < kb; ++k) {
                            s += a_row[k] * b_row[k];
                        }
                        acc_tile[static_cast<size_t>(m) * jb + n] += s;
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

} // namespace ffn
