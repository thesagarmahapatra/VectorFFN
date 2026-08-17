// benchmark_main.cpp — drives correctness checks + timing across every
// optimization level, so you get one table you can paste into a results
// section instead of five disconnected micro-benchmarks.
//
// Usage:
//   ./ffn_bench [seq_len] [d_model] [d_ff] [iters]
// Defaults are small on purpose so this runs quickly under QEMU emulation
// during development; pass realistic LLaMA/Mistral-scale dims
// (e.g. ./ffn_bench 128 4096 14336 5) when benchmarking on real M1/Snapdragon
// hardware — do NOT report the small-shape numbers as your resume figures,
// they don't reflect real cache/bandwidth pressure at production FFN sizes.

#include "common.h"
#include "gemm_naive.h"
#include "gemm_tiled.h"
#include "gemm_neon.h"
#include "silu.h"
#include "fused_ffn.h"
#include "quantize.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FFN_HAVE_ACCELERATE
#include <Accelerate/Accelerate.h>
#elif defined(FFN_HAVE_BLAS)
#include <cblas.h>
#endif

#if defined(__linux__) || defined(__ANDROID__)
#include <sched.h>
#include <unistd.h>

static void print_cpu_topology() {
    int cur_cpu = sched_getcpu();
    std::printf("Host OS: Linux/Android | Running on Core: #%d\n", cur_cpu);
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        std::printf("Active CPU affinity mask: [");
        for (int i = 0; i < 16; ++i) {
            if (CPU_ISSET(i, &cpuset)) std::printf(" %d", i);
        }
        std::printf(" ]\n");
    }
}

static void apply_cpu_affinity(const std::string& mode) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    if (mode == "prime" || mode == "x3") {
        // Snapdragon 8 Gen 2: Core 7 is Cortex-X3 Prime Core (3.36 GHz)
        CPU_SET(7, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        std::printf("Affinity set: Cortex-X3 Prime Core (#7)\n");
    } else if (mode == "perf" || mode == "big") {
        // Snapdragon 8 Gen 2: Cores 3, 4 (A715) + 5, 6 (A710) + 7 (Cortex-X3)
        for (int c = 3; c <= 7; ++c) CPU_SET(c, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        std::printf("Affinity set: Performance Cluster (Cores 3-7: Cortex-X3 + A715/A710)\n");
    } else if (mode == "little" || mode == "eff") {
        // Snapdragon 8 Gen 2: Cores 0, 1, 2 (Cortex-A510)
        for (int c = 0; c <= 2; ++c) CPU_SET(c, &cpuset);
        sched_setaffinity(0, sizeof(cpuset), &cpuset);
        std::printf("Affinity set: Efficiency Cluster (Cores 0-2: Cortex-A510)\n");
    }
}
#else
static void print_cpu_topology() {}
static void apply_cpu_affinity(const std::string&) {}
#endif

using namespace ffn;

static void print_row(const char* name, double ms, double gflops, const char* err) {
    std::printf("  %-30s %10.3f ms   %8.2f GFLOP/s   %s\n", name, ms, gflops, err);
}

int main(int argc, char** argv) {
    int seq_len = 32, d_model = 512, d_ff = 2048, iters = 3;
    std::string affinity_mode = "all";
    if (argc > 1) seq_len = std::atoi(argv[1]);
    if (argc > 2) d_model = std::atoi(argv[2]);
    if (argc > 3) d_ff    = std::atoi(argv[3]);
    if (argc > 4) iters   = std::atoi(argv[4]);
    if (argc > 5) affinity_mode = argv[5];

    if (affinity_mode != "all") {
        apply_cpu_affinity(affinity_mode);
    }

    std::printf("FFN engine benchmark — seq_len=%d d_model=%d d_ff=%d iters=%d\n",
                seq_len, d_model, d_ff, iters);
    print_cpu_topology();
#ifdef _OPENMP
    std::printf("OpenMP: enabled, max_threads=%d\n", omp_get_max_threads());
#else
    std::printf("OpenMP: NOT enabled in this build\n");
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    std::printf("ISA: NEON + dot-product extension (vdotq_s32 fast path active)\n");
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    std::printf("ISA: NEON, NO dot-product extension (INT8 uses vmull_s8 fallback)\n");
#else
    std::printf("ISA: no NEON — running portable scalar fallbacks\n");
#endif
    std::printf("\n");

    Timer timer;

    // =========================================================================
    // Part 1: standalone GEMM levels (M=seq_len, K=d_model, N=d_ff)
    // =========================================================================
    {
        const int M = seq_len, K = d_model, N = d_ff;
        std::printf("== GEMM levels: [%d,%d] x [%d,%d] ==\n", M, K, K, N);

        AlignedBuffer<float> A(static_cast<size_t>(M) * K);
        AlignedBuffer<float> B(static_cast<size_t>(K) * N);   // standard layout, for naive
        AlignedBuffer<float> B_T(static_cast<size_t>(N) * K); // transposed, for tiled/neon
        A.fill_random(-1.0f, 1.0f, 1);
        B.fill_random(-1.0f, 1.0f, 2);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n)
                B_T[static_cast<size_t>(n) * K + k] = B[static_cast<size_t>(k) * N + n];

        AlignedBuffer<float> C_ref(static_cast<size_t>(M) * N);
        AlignedBuffer<float> C_tmp(static_cast<size_t>(M) * N);

        const double flops = 2.0 * M * N * K; // one FMA = 2 FLOPs

        timer.start();
        for (int it = 0; it < iters; ++it) gemm_naive_ijk(A.data(), B.data(), C_ref.data(), M, N, K);
        double ms_naive = timer.stop_ms() / iters;
        print_row("L0 naive (i-j-k)", ms_naive, flops / (ms_naive * 1e6), "ref");

        timer.start();
        for (int it = 0; it < iters; ++it) gemm_tiled(A.data(), B_T.data(), C_tmp.data(), M, N, K);
        double ms_tiled = timer.stop_ms() / iters;
        auto err_tiled = compare(C_ref.data(), C_tmp.data(), static_cast<size_t>(M) * N);
        char buf1[64]; std::snprintf(buf1, sizeof(buf1), "nrmse=%.2e", err_tiled.nrmse);
        print_row("L1 tiled (B transposed)", ms_tiled, flops / (ms_tiled * 1e6), buf1);

        timer.start();
        for (int it = 0; it < iters; ++it) gemm_neon(A.data(), B_T.data(), C_tmp.data(), M, N, K);
        double ms_neon = timer.stop_ms() / iters;
        auto err_neon = compare(C_ref.data(), C_tmp.data(), static_cast<size_t>(M) * N);
        char buf2[64]; std::snprintf(buf2, sizeof(buf2), "nrmse=%.2e", err_neon.nrmse);
        print_row("L2 NEON (vfmaq_f32, 4-wide)", ms_neon, flops / (ms_neon * 1e6), buf2);

#if defined(FFN_HAVE_ACCELERATE) || defined(FFN_HAVE_BLAS)
        AlignedBuffer<float> C_cblas(static_cast<size_t>(M) * N);
        timer.start();
        for (int it = 0; it < iters; ++it) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        M, N, K,
                        1.0f, A.data(), K,
                        B_T.data(), K,
                        0.0f, C_cblas.data(), N);
        }
        double ms_cblas = timer.stop_ms() / iters;
        auto err_cblas = compare(C_ref.data(), C_cblas.data(), static_cast<size_t>(M) * N);
        char buf_cblas[64]; std::snprintf(buf_cblas, sizeof(buf_cblas), "nrmse=%.2e", err_cblas.nrmse);
#if defined(FFN_HAVE_ACCELERATE)
        const char* blas_label = "Apple Accelerate (cblas)";
#else
        const char* blas_label = "Vendor BLAS (cblas_sgemm)";
#endif
        print_row(blas_label, ms_cblas, flops / (ms_cblas * 1e6), buf_cblas);
#endif

        std::printf("  -> L1 vs L0 speedup: %.2fx   L2 NEON vs L0 speedup: %.2fx\n",
                    ms_naive / ms_tiled, ms_naive / ms_neon);
#if defined(FFN_HAVE_ACCELERATE) || defined(FFN_HAVE_BLAS)
        std::printf("  -> BLAS vs L0: %.2fx   L2 NEON vs BLAS: %.2fx\n",
                    ms_naive / ms_cblas, ms_cblas / ms_neon);
#endif
        std::printf("\n");

        // -------------------------------------------------------------------
        // Level 5: INT8 quantized GEMM
        // -------------------------------------------------------------------
        AlignedBuffer<int8_t> Aq(static_cast<size_t>(M) * K);
        AlignedBuffer<int8_t> Bq_T(static_cast<size_t>(N) * K);
        std::vector<float> w_scale(N);
        float act_scale = quantize_activations_per_tensor(A.data(), Aq.data(), M, K);
        quantize_weights_per_channel(B_T.data(), Bq_T.data(), w_scale.data(), N, K);

        AlignedBuffer<float> C_int8(static_cast<size_t>(M) * N);
        timer.start();
        for (int it = 0; it < iters; ++it)
            gemm_int8(Aq.data(), Bq_T.data(), C_int8.data(), M, N, K, act_scale, w_scale.data());
        double ms_int8 = timer.stop_ms() / iters;
        auto err_int8 = compare(C_ref.data(), C_int8.data(), static_cast<size_t>(M) * N);
        char buf3[96];
        std::snprintf(buf3, sizeof(buf3), "nrmse=%.4f (%.2f%%)", err_int8.nrmse, err_int8.nrmse * 100.0);
        print_row("L5 INT8 (vdotq_s32)", ms_int8, flops / (ms_int8 * 1e6), buf3);
        std::printf("  -> L5 vs L0 speedup: %.2fx\n", ms_naive / ms_int8);
        std::printf("  NOTE 1: NRMSE (RMS error / RMS signal) is reported, not max_rel_err —\n");
        std::printf("  max relative error on GEMM output is a misleading metric because any\n");
        std::printf("  near-zero true value inflates it arbitrarily even when absolute error\n");
        std::printf("  is tiny. NRMSE is the metric to actually put on a resume.\n");
        std::printf("  NOTE 2: this number is for random uniform [-1,1] data, NOT representative\n");
        std::printf("  of real trained-weight distributions. Re-measure on real weights.\n\n");
    }

    // =========================================================================
    // Part 2: full FFN — unfused (3 separate GEMMs) vs fused (register fusion)
    // =========================================================================
    {
        FFNShape shape{seq_len, d_model, d_ff};
        std::printf("== Full FFN: unfused vs register-fused ==\n");

        AlignedBuffer<float> X(static_cast<size_t>(seq_len) * d_model);
        AlignedBuffer<float> Wg_T(static_cast<size_t>(d_ff) * d_model);
        AlignedBuffer<float> Wu_T(static_cast<size_t>(d_ff) * d_model);
        AlignedBuffer<float> Wd_T(static_cast<size_t>(d_model) * d_ff);
        X.fill_random(-1.0f, 1.0f, 10);
        Wg_T.fill_random(-0.1f, 0.1f, 11);
        Wu_T.fill_random(-0.1f, 0.1f, 12);
        Wd_T.fill_random(-0.1f, 0.1f, 13);

        FFNWeights W{Wg_T.data(), Wu_T.data(), Wd_T.data()};

        AlignedBuffer<float> out_naive(static_cast<size_t>(seq_len) * d_model);
        AlignedBuffer<float> out_unfused(static_cast<size_t>(seq_len) * d_model);
        AlignedBuffer<float> out_fused(static_cast<size_t>(seq_len) * d_model);

        int naive_iters = (static_cast<double>(seq_len) * d_model * d_ff > 50000000.0) ? 1 : iters;
        timer.start();
        for (int it = 0; it < naive_iters; ++it) naive_ffn_forward(X.data(), W, out_naive.data(), shape);
        double ms_naive_ffn = timer.stop_ms() / naive_iters;

        timer.start();
        for (int it = 0; it < iters; ++it) unfused_ffn_forward(X.data(), W, out_unfused.data(), shape);
        double ms_unfused = timer.stop_ms() / iters;

        // 3 GEMMs: gate (X*Wg^T), up (X*Wu^T), down (h*Wd^T) — each 2*M*K*N FLOPs
        const double ffn_flops = 6.0 * seq_len * d_model * d_ff;

#if defined(FFN_HAVE_ACCELERATE) || defined(FFN_HAVE_BLAS)
        AlignedBuffer<float> out_cblas(static_cast<size_t>(seq_len) * d_model);
        AlignedBuffer<float> gate_cblas(static_cast<size_t>(seq_len) * d_ff);
        AlignedBuffer<float> up_cblas(static_cast<size_t>(seq_len) * d_ff);
        AlignedBuffer<float> h_cblas(static_cast<size_t>(seq_len) * d_ff);

        timer.start();
        for (int it = 0; it < iters; ++it) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, d_ff, d_model,
                        1.0f, X.data(), d_model,
                        W.Wg_T, d_model,
                        0.0f, gate_cblas.data(), d_ff);

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, d_ff, d_model,
                        1.0f, X.data(), d_model,
                        W.Wu_T, d_model,
                        0.0f, up_cblas.data(), d_ff);

            const int total_act = seq_len * d_ff;
            silu_inplace(gate_cblas.data(), h_cblas.data(), total_act);
            for (int i = 0; i < total_act; ++i) {
                h_cblas[i] *= up_cblas[i];
            }

            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, d_model, d_ff,
                        1.0f, h_cblas.data(), d_ff,
                        W.Wd_T, d_ff,
                        0.0f, out_cblas.data(), d_model);
        }
        double ms_cblas_ffn = timer.stop_ms() / iters;
        auto err_cblas_ffn = compare(out_unfused.data(), out_cblas.data(), static_cast<size_t>(seq_len) * d_model);
        char buf_cblas_ffn[64]; std::snprintf(buf_cblas_ffn, sizeof(buf_cblas_ffn), "nrmse=%.2e", err_cblas_ffn.nrmse);
#if defined(FFN_HAVE_ACCELERATE)
        const char* ffn_blas_label = "Unfused (3x Accelerate + DRAM)";
#else
        const char* ffn_blas_label = "Unfused (3x OpenBLAS + DRAM)";
#endif
        print_row(ffn_blas_label, ms_cblas_ffn, ffn_flops / (ms_cblas_ffn * 1e6), buf_cblas_ffn);
#endif

        timer.start();
        for (int it = 0; it < iters; ++it)
            fused_ffn_forward(X.data(), W, out_fused.data(), shape, /*dynamic=*/true);
        double ms_fused_dyn = timer.stop_ms() / iters;

        timer.start();
        for (int it = 0; it < iters; ++it)
            fused_ffn_forward(X.data(), W, out_fused.data(), shape, /*dynamic=*/false);
        double ms_fused_static = timer.stop_ms() / iters;

        print_row("Naive scalar (3x GEMM)", ms_naive_ffn, ffn_flops / (ms_naive_ffn * 1e6), "ref");
        print_row("Unfused (3x NEON + DRAM h)", ms_unfused, ffn_flops / (ms_unfused * 1e6), "ref");

        auto err = compare(out_unfused.data(), out_fused.data(), static_cast<size_t>(seq_len) * d_model);
        char buf[64]; std::snprintf(buf, sizeof(buf), "nrmse=%.2e", err.nrmse);

        print_row("Fused, schedule(dynamic)", ms_fused_dyn, ffn_flops / (ms_fused_dyn * 1e6), buf);
        print_row("Fused, schedule(static)", ms_fused_static, ffn_flops / (ms_fused_static * 1e6), buf);
        std::printf("  -> overall speedup (fused vs naive): %.2fx\n", ms_naive_ffn / ms_fused_dyn);
        std::printf("  -> fusion speedup (dynamic vs unfused): %.2fx   (static): %.2fx\n",
                    ms_unfused / ms_fused_dyn, ms_unfused / ms_fused_static);
#if defined(FFN_HAVE_ACCELERATE) || defined(FFN_HAVE_BLAS)
        std::printf("  -> fusion vs 3x BLAS pipeline: %.2fx\n", ms_cblas_ffn / ms_fused_dyn);
#endif
        std::printf("  NOTE: static vs dynamic scheduling only diverges meaningfully on\n");
        std::printf("  heterogeneous P/E cores (M1, Snapdragon) with >1 thread and enough\n");
        std::printf("  tokens to actually need load balancing. On this sandbox's uniform\n");
        std::printf("  x86 cores the two will look similar — re-run this exact binary on\n");
        std::printf("  your M1/Tab S9 to see the real divergence.\n");
    }

    return 0;
}

