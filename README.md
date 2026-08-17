# ⚡ VectorFFN

**High-performance on-device LLM inference engine for ARM64 SIMD architectures.**

Bare-metal C++17 implementation of fused SwiGLU micro-kernels targeting the dominant compute bottleneck in modern transformer models (LLaMA-3, Mistral, Gemma, DeepSeek):

$$\text{FFN}(x) = \left(\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})\right) \cdot W_{\text{down}}$$

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![SIMD](https://img.shields.io/badge/SIMD-ARM%20NEON%20%7C%20dotprod%20%7C%208--Wide-red.svg)](#)
[![ISA](https://img.shields.io/badge/ISA-AArch64%20%7C%20ARMv8.5--A%20%7C%20ARMv9-green.svg)](#)
[![Quantization](https://img.shields.io/badge/Quantization-INT8%20(vdotq__s32)-yellow.svg)](#)
[![Multi-Threading](https://img.shields.io/badge/Threading-OpenMP-orange.svg)](#)
[![Platforms](https://img.shields.io/badge/Platforms-Snapdragon%20%7C%20Apple%20Silicon%20%7C%20Graviton-purple.svg)](#)
[![Target Models](https://img.shields.io/badge/Models-LLaMA--3%20%7C%20Mistral%20%7C%20Gemma-informational.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## Why This Exists

In modern LLMs, the FFN layers account for ~65% of model parameters and dominate per-token latency. Standard inference frameworks dispatch Gate, Up, and Down projections as separate GEMM calls, writing large intermediate activation matrices to DRAM between each step:

```
Standard Unfused Pipeline:
                                                   DRAM             DRAM             DRAM
  x -----> [Gate GEMM] -----> write gate --------> read ----+
  x -----> [Up GEMM]   -----> write up   --------> read ----+--> SiLU + multiply --> write h --> read h --> [Down GEMM] --> output
                                                             3 full [seq, d_ff] DRAM roundtrips

VectorFFN Fused Pipeline:
                                 All in NEON registers (v0..v7)
  x -----> [Gate + Up dot products] --> [SiLU] --> [multiply] --> accumulate into Down --> output
                                 Zero intermediate DRAM writes
```

VectorFFN eliminates this overhead by fusing the entire SwiGLU pipeline directly within ARM NEON vector registers.

---

## Optimization Levels

| Level | Technique | What It Does |
| :---: | :--- | :--- |
| L0 | Naive GEMM | Baseline triple-loop reference |
| L1 | Cache-Tiled GEMM | Offline weight transposition ($B^T$) + 3D loop blocking for cache locality |
| L2 | **NEON SIMD Microkernel** | 8-wide register-blocked FMA (`vfmaq_f32`) with 8 vector accumulators |
| L3 | Vectorized SiLU | Polynomial approximation with IEEE-754 exponent bit manipulation |
| L4 | **Fused SwiGLU FFN** | In-register fusion of Gate + Up + SiLU + Hadamard (>80% DRAM traffic eliminated) |
| L5 | **INT8 Quantization** | Hardware `vdotq_s32` dot-product instructions with per-channel scaling |

---

## How the Optimizations Work

### Matrix Multiplication: Naive vs Tiled vs SIMD

```
L0 Naive GEMM: A[M,K] x B[K,N]                L1 Tiled GEMM: A[M,K] x B_T[N,K]
                                                (weights pre-transposed offline)
  for i in M:                                    for tile_m in M (step MT):
    for j in N:                                    for tile_n in N (step NT):
      for k in K:                                    for tile_k in K (step KT):
        C[i][j] += A[i][k] * B[k][j]                  // inner loop on cache-resident block
               stride-N jumps ^^^^^                    // both A and B_T are contiguous now
               = cache miss every iteration            // accumulator stays in L1 until K done
```

### SIMD: Scalar vs 8-Wide NEON Microkernel

```
Scalar (1 multiply per cycle):         NEON 8-wide (32 multiplies per cycle):

  sum += a[k] * b[k]                     acc0 = vfmaq_f32(acc0, va, vb0)   // 4 FMAs
  sum += a[k+1] * b[k+1]                 acc1 = vfmaq_f32(acc1, va, vb1)   // 4 FMAs
  sum += a[k+2] * b[k+2]                 acc2 = vfmaq_f32(acc2, va, vb2)   // 4 FMAs
  sum += a[k+3] * b[k+3]                 acc3 = vfmaq_f32(acc3, va, vb3)   // 4 FMAs
  ...                                    acc4 = vfmaq_f32(acc4, va, vb4)   // 4 FMAs
  (1 result per 4 cycles)                acc5 = vfmaq_f32(acc5, va, vb5)   // 4 FMAs
                                         acc6 = vfmaq_f32(acc6, va, vb6)   // 4 FMAs
                                         acc7 = vfmaq_f32(acc7, va, vb7)   // 4 FMAs
                                         (8 dot products from 1 A-row load)
```

Uses 17 of 32 ARM NEON vector registers: 8 accumulators + 1 shared A + 8 B rows. No register spilling.

### Register Fusion: Why It Matters

```
Without fusion (standard ML frameworks):         With fusion (VectorFFN):

  gate[seq, d_ff] = x @ W_gate   --> DRAM write    for each output column:
  up[seq, d_ff]   = x @ W_up     --> DRAM write      gate_4 = dot(x, W_gate_col)  // stays in v0
  h = SiLU(gate) * up            --> DRAM read x2     up_4   = dot(x, W_up_col)    // stays in v1
  out = h @ W_down                --> DRAM read        h_4    = SiLU(gate_4) * up_4  // stays in v2
                                                       accumulate h_4 into down proj
  Memory: 3 full [seq, d_ff] tensors
  written and read back from DRAM                   Memory: zero intermediate tensors in DRAM
```

### Heterogeneous Core Scheduling

```
Snapdragon 8 Gen 2 (1+4+3 topology):      Apple M1 (4+4 topology):

  Core 7:  Cortex-X3  @ 3.36 GHz (fast)     Cores 0-3: Firestorm @ 3.20 GHz (fast)
  Core 3-6: A715/A710 @ 2.80 GHz (fast)     Cores 4-7: Icestorm  @ 2.06 GHz (slow)
  Core 0-2: A510      @ 2.00 GHz (slow)

  Static scheduling across all cores:       Static scheduling across all cores:
    Fast cores finish early, wait             Same problem: P-cores idle at barrier
    at OpenMP barrier for slow cores          waiting for E-cores to finish

  Solution: pin to performance cluster      Solution: OMP_NUM_THREADS=4 (P-cores only)
    --> 2.08x faster than naive 8-core        --> 1.45x faster than naive 8-core
```

---

## Benchmarks

Tested at LLaMA-3 / Mistral-7B production scale (M=128, K=4096, N=14336).

### Qualcomm Snapdragon 8 Gen 2 (Galaxy Tab S9)

| Kernel | Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| Naive GEMM | 68,617 ms | 0.22 GFLOP/s | 1x |
| Cache-Tiled | 5,112 ms | 2.94 GFLOP/s | 13.4x |
| NEON FP32 (8x1) | 812 ms | 18.50 GFLOP/s | 84.5x |
| **INT8 (vdotq_s32)** | **360 ms** | **41.74 GFLOP/s** | **190.5x** |
| Fused SwiGLU FFN | 1,691 ms | 26.67 GFLOP/s | 1.43x vs unfused |

### Apple Silicon M1 (MacBook Air)

| Kernel | Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| Naive GEMM | 27,504 ms | 0.55 GFLOP/s | 1x |
| Cache-Tiled | 5,199 ms | 2.89 GFLOP/s | 5.3x |
| NEON FP32 (8x1) | 742 ms | 20.27 GFLOP/s | 37.1x |
| **INT8 (vdotq_s32)** | **378 ms** | **39.74 GFLOP/s** | **72.7x** |
| Fused SwiGLU (4 P-Cores) | 1,354 ms | 33.31 GFLOP/s | 1.48x vs unfused |
| *Apple Accelerate (AMX)* | *28 ms* | *529 GFLOP/s* | *969x (dedicated HW)* |

All optimized kernels maintain NRMSE < 0.6% against FP32 ground truth.

---

## Build

### macOS (Apple Silicon)
```bash
brew install cmake libomp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
./ffn_bench 128 4096 14336 2
```

### Android / Snapdragon (Cross-compile via NDK)
```bash
NDK_CC=$(find /opt/homebrew/Caskroom/android-ndk -name "aarch64-linux-android*clang++" | head -1)
$NDK_CC -O3 -std=c++17 -march=armv8.5-a+dotprod \
  -fopenmp -static-openmp -static-libstdc++ \
  -Iinclude src/*.cpp -o ffn_bench_android

adb push ffn_bench_android /data/local/tmp/
adb shell "/data/local/tmp/ffn_bench_android 128 4096 14336 2 perf"
```

### Android (Termux, on-device)
```bash
pkg install clang cmake libomp openblas
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFFN_FORCE_ARM_MARCH=armv8.5-a+dotprod
cmake --build . -j
./ffn_bench 128 4096 14336 2 perf
```

---

## Repository Structure

```
VectorFFN/
├── CMakeLists.txt          # Build config (ISA detection, OpenMP, BLAS backends)
├── include/
│   ├── common.h            # Aligned buffers, timer, NRMSE validator
│   ├── gemm_naive.h        # L0: Scalar baseline
│   ├── gemm_tiled.h        # L1: Cache-tiled GEMM
│   ├── gemm_neon.h         # L2: NEON SIMD microkernels
│   ├── silu.h              # L3: Vectorized SiLU
│   ├── fused_ffn.h         # L4: Fused SwiGLU pipeline
│   └── quantize.h          # L5: INT8 quantized GEMM
├── src/                    # Implementations + benchmark harness
├── LICENSE
└── README.md
```

## License
[MIT](LICENSE)
