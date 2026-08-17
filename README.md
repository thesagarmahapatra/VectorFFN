# ⚡ VectorFFN

**High-performance on-device LLM inference engine for ARM64 SIMD architectures.**

Bare-metal C++17 implementation of fused SwiGLU micro-kernels targeting the dominant compute bottleneck in modern transformer models (LLaMA-3, Mistral, Gemma, DeepSeek):

$$\text{FFN}(x) = \left(\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})\right) \cdot W_{\text{down}}$$

[![C++17](https://img.shields.io/badge/C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![SIMD](https://img.shields.io/badge/SIMD-ARM%20NEON%20%7C%20dotprod-red.svg)](#)
[![Platforms](https://img.shields.io/badge/Qualcomm%20Snapdragon%20%7C%20Apple%20Silicon%20%7C%20AWS%20Graviton-purple.svg)](#)

---

## Key Ideas

In modern LLMs, the FFN layers account for ~65% of model parameters and dominate per-token latency. Standard inference frameworks dispatch Gate, Up, and Down projections as separate GEMM calls, writing large intermediate activation matrices to DRAM between each step.

VectorFFN eliminates this overhead by **fusing the entire SwiGLU pipeline directly within ARM NEON vector registers** — Gate projection, Up projection, polynomial SiLU, and Hadamard product all execute in-register with zero intermediate DRAM roundtrips.

---

## Optimization Levels

| Level | Technique | What It Does |
| :---: | :--- | :--- |
| L0 | Naive GEMM | Baseline triple-loop reference |
| L1 | Cache-Tiled GEMM | Offline weight transposition ($B^T$) + 3D loop blocking for L1/L2 cache locality |
| L2 | NEON SIMD Microkernel | 8-wide register-blocked FMA (`vfmaq_f32`) with 8 vector accumulators |
| L3 | Vectorized SiLU | Polynomial approximation with IEEE-754 exponent bit-manipulation |
| L4 | **Fused SwiGLU FFN** | In-register fusion of Gate + Up + SiLU + Hadamard (>80% DRAM traffic eliminated) |
| L5 | **INT8 Quantization** | Hardware `vdotq_s32` dot-product instructions with per-channel scaling |

---

## Benchmarks

Tested at **LLaMA-3 / Mistral-7B production scale** ($M=128, K=4096, N=14336$).

### Qualcomm Snapdragon 8 Gen 2 (Galaxy Tab S9)

| Kernel | Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| Naive GEMM | 68,617 ms | 0.22 GFLOP/s | 1× |
| Cache-Tiled | 5,112 ms | 2.94 GFLOP/s | 13.4× |
| NEON FP32 (8×1) | 812 ms | 18.50 GFLOP/s | 84.5× |
| **INT8 (`vdotq_s32`)** | **360 ms** | **41.74 GFLOP/s** | **190.5×** |
| Fused SwiGLU FFN | **1,691 ms** | **26.67 GFLOP/s** | 1.43× vs Unfused |

**Cluster scheduling:** Pinning threads to the Performance Cluster (Cortex-X3 + A715/A710) and excluding A510 efficiency cores yields a **2.08× latency reduction** over naive 8-core static scheduling by eliminating OpenMP barrier stalls.

### Apple Silicon M1 (MacBook Air)

| Kernel | Latency | Throughput | Speedup |
| :--- | :---: | :---: | :---: |
| Naive GEMM | 27,504 ms | 0.55 GFLOP/s | 1× |
| Cache-Tiled | 5,199 ms | 2.89 GFLOP/s | 5.3× |
| NEON FP32 (8×1) | 742 ms | 20.27 GFLOP/s | 37.1× |
| **INT8 (`vdotq_s32`)** | **378 ms** | **39.74 GFLOP/s** | **72.7×** |
| Fused SwiGLU FFN (4 P-Cores) | **1,354 ms** | **33.31 GFLOP/s** | 1.48× vs Unfused |
| *Apple Accelerate (AMX)* | *28 ms* | *529.65 GFLOP/s* | *969× (dedicated HW)* |

**Thread scaling:** Running on 4 Firestorm P-cores only outperforms all 8 cores (4P+4E) by 1.45× — adding Icestorm E-cores introduces barrier synchronization overhead that exceeds their compute contribution.

> **Numerical fidelity:** All optimized kernels maintain NRMSE < 0.6% against FP32 ground truth. INT8 quantization uses per-channel weight scaling.

---

## How It Works

**1. Weight Transposition ($B^T$):** Pre-transposes weight matrices offline so both operands stream contiguously through L1 cache via unit-stride `vld1q_f32` loads.

**2. 3D Cache Blocking:** Tiles the GEMM along M, N, K dimensions to keep working sets in L1/L2. Accumulators commit to DRAM only once per full K-sweep.

**3. 8×1 Register Microkernel:** Processes 8 rows of $B^T$ against 1 row of $A$ using 8 NEON vector accumulators (17 of 32 available `v` registers), delivering 8 FMAs per A-row load.

**4. In-Register Fusion:** Gate and Up dot-products, polynomial SiLU activation, and Hadamard product all execute within the vector register file — intermediate activations never touch DRAM.

**5. Asymmetric Core Scheduling:** On heterogeneous big.LITTLE CPUs, pins OpenMP threads to the performance cluster via `sched_setaffinity` to eliminate barrier tail latency from slow efficiency cores.

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
