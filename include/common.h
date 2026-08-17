#pragma once
// common.h — shared types, aligned allocation, timing utilities.
//
// Design note: every GEMM/FFN variant in this project operates on the same
// AlignedBuffer type so that benchmark_main.cpp can swap kernels without
// touching data layout. Row-major throughout: A is [M,K], B is [K,N] (already
// transposed to [N,K] for the "B_T" kernels — see gemm_neon.h for why).

#include <cstdint>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <memory>
#include <random>
#include <stdexcept>

namespace ffn {

// ---------------------------------------------------------------------------
// Aligned buffer: NEON vld1q_f32 / vld1q_s8 want 16-byte alignment to avoid
// splitting loads across cache lines. We over-align to 64B (a full M1 cache
// line) so tiles never straddle a line boundary either.
// ---------------------------------------------------------------------------
template <typename T>
class AlignedBuffer {
public:
    explicit AlignedBuffer(size_t count) : count_(count) {
        constexpr size_t kAlign = 64;
        size_t bytes = ((count * sizeof(T) + kAlign - 1) / kAlign) * kAlign;
        if (bytes == 0) bytes = kAlign;
        void* p = nullptr;
        if (posix_memalign(&p, kAlign, bytes) != 0 || p == nullptr) {
            throw std::bad_alloc();
        }
        data_ = static_cast<T*>(p);
    }
    ~AlignedBuffer() { std::free(data_); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    T* data() { return data_; }
    const T* data() const { return data_; }
    T& operator[](size_t i) { return data_[i]; }
    const T& operator[](size_t i) const { return data_[i]; }
    size_t size() const { return count_; }

    void fill_random(float lo = -1.0f, float hi = 1.0f, uint32_t seed = 42) {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> dist(lo, hi);
        for (size_t i = 0; i < count_; ++i) data_[i] = static_cast<T>(dist(rng));
    }

    void zero() { std::memset(data_, 0, count_ * sizeof(T)); }

private:
    T* data_ = nullptr;
    size_t count_ = 0;
};

// ---------------------------------------------------------------------------
// FFN problem shape. Mirrors a single decoder-layer FFN call:
//   x:      [seq_len, d_model]
//   W_gate: [d_model, d_ff]   (stored transposed as [d_ff, d_model] — see below)
//   W_up:   [d_model, d_ff]   (stored transposed as [d_ff, d_model])
//   W_down: [d_ff, d_model]   (stored transposed as [d_model, d_ff])
//
// Why transposed weight storage: every GEMM here computes
//   out[m,n] = sum_k A[m,k] * B_T[n,k]
// i.e. both operands are walked row-major with unit stride. This is the
// standard "B pre-transposed" trick that turns the classic cache-hostile
// i-k-j GEMM into something where *both* input streams are sequential in
// memory, which is what makes NEON loads (vld1q) and cache tiling actually
// pay off. Weight transposition is a one-time offline cost (like folding
// BatchNorm into Conv), not part of the hot path.
// ---------------------------------------------------------------------------
struct FFNShape {
    int seq_len;   // M
    int d_model;   // K for gate/up projections, N for down projection
    int d_ff;      // N for gate/up projections, K for down projection
};

// ---------------------------------------------------------------------------
// Timing helper
// ---------------------------------------------------------------------------
class Timer {
public:
    void start() { t0_ = std::chrono::high_resolution_clock::now(); }
    double stop_ms() {
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }
private:
    std::chrono::high_resolution_clock::time_point t0_;
};

// Numerical comparison helper for validating optimized kernels against the
// naive reference.
//
// max_rel is included for completeness but is a KNOWN-MISLEADING metric on
// GEMM outputs: any element whose true value happens to be near zero
// (routine for random/zero-mean data) makes an ordinary small absolute
// error look like an enormous relative error, because the denominator
// collapses. NRMSE (normalized root-mean-square error — RMS of the error
// divided by RMS of the reference signal) is the metric actually used to
// report quantization accuracy in practice for exactly this reason, and is
// what should go on a resume/report, not max_rel.
struct ErrStats { double max_abs; double max_rel; double mean_abs; double nrmse; };

inline ErrStats compare(const float* ref, const float* got, size_t n) {
    double max_abs = 0.0, max_rel = 0.0, sum_abs = 0.0, sum_sq = 0.0, ref_sum_sq = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double d = std::abs(static_cast<double>(ref[i]) - static_cast<double>(got[i]));
        double denom = std::max(1e-6, std::abs(static_cast<double>(ref[i])));
        max_abs = std::max(max_abs, d);
        max_rel = std::max(max_rel, d / denom);
        sum_abs += d;
        sum_sq += d * d;
        ref_sum_sq += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
    }
    double nrmse = (ref_sum_sq > 0.0)
        ? std::sqrt(sum_sq / n) / std::sqrt(ref_sum_sq / n)
        : std::sqrt(sum_sq / n);
    return {max_abs, max_rel, sum_abs / static_cast<double>(n), nrmse};
}

} // namespace ffn
