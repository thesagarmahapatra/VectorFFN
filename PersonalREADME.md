# Fused SwiGLU / Transformer FFN Inference Engine (ARM NEON, OpenMP & Qualcomm Snapdragon)

An optimized, CPU-based feed-forward network (FFN) inference engine implementing the modern LLaMA-3 / Mistral **SwiGLU** activation block:
$$\text{FFN}(x) = \left(\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})\right) \cdot W_{\text{down}}$$

Implemented from scratch in **C++17** and systematically optimized across six distinct levels: from naive unvectorized matrix multiplication to 3D cache-blocked tiling, $8\times 1$ and $4\times 1$ ARM NEON register microkernels, OpenMP multithreading, register-level kernel fusion, INT8 quantization (`vdotq_s32`), Qualcomm Snapdragon 8 Gen 2 tri-cluster core affinity, and benchmarking against vendor BLAS libraries (**Apple Accelerate `cblas_sgemm` on macOS, OpenBLAS on Android/Linux**).

---

## 🚀 Optimization Hierarchy

| Level | Implementation | Core Technique / Architectural Mechanism |
| :--- | :--- | :--- |
| **L0** | Naive GEMM | Baseline triple-loop ($i\text{-}j\text{-}k$) exhibiting severe stride-$N$ cache line thrashing. |
| **L1** | Cache-Tiled GEMM | Pre-transposed offline weights ($B^T$) + 3-level loop blocking ($M_T, N_T, K_T$) for L1/L2 cache locality and single DRAM write-back. |
| **L2** | NEON FP32 GEMM | $8\times 1$ & $4\times 1$ register blocking with `vfmaq_f32` FMA and `vaddvq_f32` pairwise horizontal reductions. |
| **L3** | Vectorized SiLU | Fast polynomial approximation with IEEE-754 exponent bit-manipulation range-reduction ($e^x$). |
| **L4** | **Fused SwiGLU FFN** | Fuses Gate + Up projections, SiLU, and Hadamard products directly in NEON registers (**zero intermediate DRAM roundtrips**). |
| **L5** | **INT8 Quantized GEMM** | Per-channel weight / per-tensor activation scaling with ARMv8.2-A `vdotq_s32` 4-way dot-product SIMD instructions. |
| **Ref** | **Vendor BLAS Reference** | External gold-standard baseline using `cblas_sgemm` (Apple Accelerate on macOS / OpenBLAS on Linux/Android). |

---

## 📱 Qualcomm Snapdragon 8 Gen 2 Architecture (Galaxy Tab S9)

The Samsung Galaxy Tab S9 is powered by the **Qualcomm Snapdragon 8 Gen 2 (SM8550)** featuring a **1+4+3 tri-cluster CPU topology**:

```
Snapdragon 8 Gen 2 CPU Topology:
├── Core 7: 1x Cortex-X3 Prime Core (3.36 GHz, 1MB L2, 8MB L3) -> Peak Single-Thread Compute
├── Cores 3-6: 4x Performance Cores (2.80 GHz)
│   ├── 2x Cortex-A715 (Cores 3, 4)
│   └── 2x Cortex-A710 (Cores 5, 6)
└── Cores 0-2: 3x Cortex-A510 Efficiency Cores (2.00 GHz)
```

### Qualcomm-Specific Optimizations Implemented:
1. **Tri-Cluster Core Affinity (`sched_setaffinity`)**:
   - Running multithreaded GEMM across all 8 cores naively causes synchronization stalls because the A510 efficiency cores run at lower clock rates and narrower SIMD throughput.
   - We added runtime cluster pinning: `--affinity [perf|prime|little|all]`. Pinning worker threads to the 5 High-Performance cores (Cores 3-7) eliminates the barrier tail latency.
2. **8-Wide NEON Microkernel (`microkernel_8x1` & `dot8`)**:
   - Leverages 8 vector accumulators (`v0`..`v7`) to fully saturate the dual 128-bit FMA pipelines and wide decode/dispatch of the Cortex-X3 and Cortex-A715 cores.
3. **Cross-Platform CBLAS Interface**:
   - Automatically binds to `libopenblas` / `libcblas` on Android/Linux to provide the exact vendor BLAS comparison equivalent to Apple Accelerate on macOS.

---

## 📊 Live Benchmark Results

### 1. Qualcomm Snapdragon 8 Gen 2 (Samsung Galaxy Tab S9)

#### A. Full Production LLaMA-3 / Mistral-7B Scale ($M=128, K=4096, N=14336$)
*Tested on Performance Cluster (Cores 3–7: Cortex-X3 + Cortex-A715/A710):*
| Optimization Level / Kernel | Latency | Throughput | Numerical Fidelity | Speedup vs Naive |
| :--- | :---: | :---: | :---: | :---: |
| **L0: Naive GEMM ($i\text{-}j\text{-}k$)** | 68,617.27 ms | 0.22 GFLOP/s | Reference | $1.0\times$ |
| **L1: Cache-Tiled ($B^T$)** | 5,112.13 ms | 2.94 GFLOP/s | $\text{NRMSE}=1.19\times 10^{-6}$ | **$13.42\times$** |
| **L2: NEON FP32 ($8\times 1$)** | 812.37 ms | 18.50 GFLOP/s | $\text{NRMSE}=1.16\times 10^{-6}$ | **$84.47\times$** |
| **L5: INT8 (`vdotq_s32`)** | **360.15 ms** | **41.74 GFLOP/s** | $\text{NRMSE}=0.56\%$ | **$190.52\times$** ($2.26\times$ vs FP32) |
| **Full FFN: Unfused (3x NEON + DRAM $h$)** | 2,424.17 ms | 18.60 GFLOP/s | Reference | $6.81\times$ |
| **Full FFN: Fused NEON (`dynamic`)** | **1,691.16 ms** | **26.67 GFLOP/s** | $\text{NRMSE}=1.40\times 10^{-6}$ | **$9.76\times$ vs Naive ($1.43\times$ vs Unfused)** |

#### B. Medium Scale ($M=64, K=2048, N=5632$) — Cluster & Scheduling Analysis
| Core Topology / Configuration | Latency | Throughput | Speedup / Scheduling Behavior |
| :--- | :---: | :---: | :--- |
| **L0 Naive GEMM** | 4,486.50 ms | 0.33 GFLOP/s | Unvectorized baseline |
| **L2 NEON FP32** | 80.66 ms | 18.30 GFLOP/s | **$55.63\times$ vs Naive** |
| **L5 INT8 (`vdotq_s32`)** | **34.18 ms** | **43.19 GFLOP/s** | **$131.25\times$ vs Naive**, 0.56% NRMSE |
| **Full FFN (All 8 Cores, Static OpenMP)** | 336.44 ms | 13.16 GFLOP/s | Stalls at OpenMP barrier waiting on A510 cores |
| **Full FFN (All 8 Cores, Dynamic OpenMP)** | 224.52 ms | 19.73 GFLOP/s | Dynamic work-stealing ($1.50\times$ faster than static) |
| **Full FFN (Performance Cluster: Cores 3–7)** | **161.58 ms** | **27.41 GFLOP/s** | **$2.08\times$ vs 8-core static**, zero barrier stalls |

---

### 2. Apple Silicon M1 (MacBook Air)

#### A. Full Production LLaMA-3 / Mistral-7B Scale ($M=128, K=4096, N=14336$)
*Tested across thread configurations (4 Performance Cores vs. 4P + 4E Cores):*
| Optimization Level / Kernel | Latency | Throughput | Numerical Fidelity | Speedup vs Naive |
| :--- | :---: | :---: | :---: | :---: |
| **L0: Naive GEMM ($i\text{-}j\text{-}k$)** | 27,503.54 ms | 0.55 GFLOP/s | Reference | $1.0\times$ |
| **L1: Cache-Tiled ($B^T$)** | 5,199.43 ms | 2.89 GFLOP/s | $\text{NRMSE}=1.19\times 10^{-6}$ | **$5.29\times$** |
| **L2: NEON FP32 ($8\times 1$)** | 741.75 ms | 20.27 GFLOP/s | $\text{NRMSE}=1.16\times 10^{-6}$ | **$37.08\times$** |
| **L5: INT8 (`vdotq_s32`)** | **378.29 ms** | **39.74 GFLOP/s** | $\text{NRMSE}=0.56\%$ | **$72.71\times$** ($1.96\times$ vs FP32) |
| *Ref: Apple Accelerate (`cblas`)* | 28.38 ms | 529.65 GFLOP/s | Reference | $969.06\times$ (AMX) |

#### Full SwiGLU FFN Pipeline (Thread Sweep: 4 P-Cores vs 8 Cores)
| Thread Configuration | Latency | Throughput | Speedup / Scheduling Behavior |
| :--- | :---: | :---: | :--- |
| **Naive Scalar (3x GEMM)** | 22,716.49 ms | 1.99 GFLOP/s | Baseline reference |
| **Unfused (3x NEON + DRAM $h$)** | 2,003.82 ms | 22.51 GFLOP/s | DRAM intermediate activation roundtrips |
| **Fused NEON (8 Cores, Static OpenMP)** | 1,965.68 ms | 22.94 GFLOP/s | Stalls at OpenMP barrier waiting on 4 E-cores |
| **Fused NEON (8 Cores, Dynamic OpenMP)** | 1,654.62 ms | 27.26 GFLOP/s | Dynamic work-stealing ($1.19\times$ faster than 8-core static) |
| **Fused NEON (4 P-Cores Only: `OMP_NUM_THREADS=4`)** | **1,353.84 ms** | **33.31 GFLOP/s** | **$16.47\times$ vs Naive ($1.48\times$ vs Unfused)**, peak efficiency |
| *Ref: Unfused (3x Accelerate + DRAM)* | 91.09 ms | 495.10 GFLOP/s | Vendor AMX baseline with DRAM traffic |

---

## 🛠️ How to Build & Benchmark

### 1. Galaxy Tab S9 (via ADB from Mac)
```bash
# Cross-compile using Android NDK
NDK_COMPILER=$(find /opt/homebrew/share/android-ndk /opt/homebrew/Caskroom/android-ndk -name "aarch64-linux-android*clang++" | grep -E "android(29|30|31|32|33|34)" | head -n 1)

$NDK_COMPILER -O3 -std=c++17 \
  -march=armv8.5-a+dotprod \
  -fopenmp -static-openmp -static-libstdc++ \
  -Iinclude src/*.cpp -o build/ffn_bench_sd8g2

# Push & Run on Tab S9
adb push build/ffn_bench_sd8g2 /data/local/tmp/
adb shell "chmod +x /data/local/tmp/ffn_bench_sd8g2 && /data/local/tmp/ffn_bench_sd8g2 128 4096 14336 2 perf"
```

### 2. Galaxy Tab S9 (Inside Termux)
```bash
# Install tools & clone
termux-setup-storage
pkg update && pkg install -y clang cmake libomp openblas git
cp -r /sdcard/Download/ffn_engine ~/ffn_engine && cd ~/ffn_engine

# Build & Run
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFFN_FORCE_ARM_MARCH=armv8.5-a+dotprod
cmake --build . -j
./ffn_bench 128 4096 14336 2 perf
```

### 3. macOS (Apple Silicon M1/M2/M3/M4)
```bash
brew install cmake libomp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Sweep threads
OMP_NUM_THREADS=4 ./ffn_bench 128 4096 14336 2
OMP_NUM_THREADS=8 ./ffn_bench 128 4096 14336 2
```

---

## 🔍 Microarchitectural Deep Dive & Design Rationale

### 1. Offline Weight Transposition ($B^T$)
Standard row-major matrix multiplication $C_{i,j} = \sum_k A_{i,k} \cdot B_{k,j}$ accesses $B$ with stride-$N$. Once $N > 16$, every inner-loop iteration accesses a different cache line, causing $100\%$ cache misses. Transposing weights offline ($B^T [N, K]$) converts both $A$ and $B^T$ into contiguous row-major streams, enabling sequential unit-stride vector loads (`vld1q_f32`) for both operands.

### 2. 3-Level Cache Tiling ($M_T, N_T, K_T$)
Rather than accumulating directly to DRAM, the outer loops partition the computation into blocks that fit within L1/L2 caches. Accumulation happens in a local cache-resident tile, ensuring that matrix $C$ is written to main memory **only once per full $K$-sweep**, drastically cutting memory bandwidth demand.

### 3. $8\times 1$ NEON Register-Blocked Microkernel
The microkernel processes 8 output rows of $B^T$ against 1 row of $A$ simultaneously using 8 NEON 128-bit vector accumulators (`v0`..`v7`). This achieves **8 FMAs per $A$-row load**, maximizing arithmetic intensity and instruction-level parallelism on wide out-of-order execution cores (Cortex-X3 / Firestorm).

### 4. Register Kernel Fusion for SwiGLU
Standard inference frameworks materialize intermediate $[M, d_{\text{ff}}]$ tensors for $\text{Gate}(x)$ and $\text{Up}(x)$ into DRAM. The fused engine computes Gate and Up projections directly into CPU vector registers, evaluates the polynomial SiLU and Hadamard product in-place, and directly accumulates into the projection stream—reducing activation DRAM traffic by **$>80\%$**.

### 5. Metric Rigor: Why NRMSE vs. Max Relative Error
Max relative error ($\max \frac{|y - \hat{y}|}{|y|}$) is mathematically unstable for GEMM validation because near-zero values at the numerical noise floor artificially inflate relative error to thousands of percent. We use **Normalized Root-Mean-Square Error (NRMSE)**:
$$\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum (y_i - \hat{y}_i)^2}}{\sqrt{\frac{1}{N}\sum y_i^2}}$$
Our INT8 kernel achieves **$<0.6\%$ NRMSE**, proving numerical fidelity for production LLM activations.

---

## 📂 Repository Structure

```
ffn_engine/
├── CMakeLists.txt         # Cross-platform build config (ISA flags, OpenMP, Accelerate, OpenBLAS)
├── include/
│   ├── common.h           # AlignedBuffer (64B cache alignment), Timer, NRMSE validator
│   ├── gemm_naive.h       # L0: Naive scalar GEMM reference
│   ├── gemm_tiled.h       # L1: Cache-tiled 3D blocking GEMM
│   ├── gemm_neon.h        # L2: NEON FP32 8-wide & 4-wide register-blocked microkernels
│   ├── silu.h             # L3: Vectorized polynomial SiLU (exp range reduction)
│   ├── fused_ffn.h        # L4: Multithreaded register-fused SwiGLU pipeline
│   └── quantize.h         # L5: INT8 per-channel quantization & vdotq_s32 GEMM
├── src/
│   ├── benchmark_main.cpp # Unified benchmark harness with CPU affinity & BLAS baselines
│   ├── gemm_naive.cpp
│   ├── gemm_tiled.cpp
│   ├── gemm_neon.cpp
│   ├── silu.cpp
│   ├── fused_ffn.cpp
│   └── quantize.cpp
└── README.md
```
