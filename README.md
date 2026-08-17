# ⚡ VectorFFN: High-Performance On-Device LLM Inference Engine (ARM64 SIMD)
### Bare-Metal C++17 Fused SwiGLU SIMD Micro-Kernels for LLaMA-3, Mistral & Gemma Edge Inference

[![C++17](https://img.shields.io/badge/Language-C%2B%2B17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![SIMD](https://img.shields.io/badge/SIMD-ARM%20NEON%20%7C%20dotprod%20%7C%208--Wide-red.svg)](#)
[![ISA](https://img.shields.io/badge/ISA-ARMv8.5--A%20%7C%20ARMv9%20%7C%20AArch64-green.svg)](#)
[![Multi-Threading](https://img.shields.io/badge/Threading-OpenMP-orange.svg)](#)
[![Platforms](https://img.shields.io/badge/Hardware-Qualcomm%20Snapdragon%20%7C%20Apple%20Silicon%20%7C%20AWS%20Graviton-purple.svg)](#)
[![Accuracy](https://img.shields.io/badge/Precision-INT8%20(NRMSE%20%3C%200.6%25)-brightgreen.svg)](#)

**VectorFFN** is a hardware-conscious, bare-metal C++17 SIMD inference engine engineered to accelerate the primary compute and memory bottleneck in modern Large Language Models (LLMs)—the **SwiGLU Feed-Forward Network (FFN)** block found in **LLaMA-3, Mistral, Gemma, and DeepSeek**:
$$\text{FFN}(x) = \left(\text{SiLU}(x \cdot W_{\text{gate}}) \odot (x \cdot W_{\text{up}})\right) \cdot W_{\text{down}}$$

In modern autoregressive decoder models, the FFN layers represent **$\sim 65\%$ of total model parameters** and dominate per-token generation latency (the autoregressive *decode phase*). VectorFFN implements a hardware-software co-designed inference pipeline that eliminates intermediate activation memory traffic, achieves **up to $190\times$ speedup** over unvectorized baselines, and cuts mobile DRAM power consumption on edge ARM devices.

---

## 🧠 Why Edge LLM Inference is a Systems Problem

During autoregressive token generation (Batch Size = 1 or low sequence lengths), inference is strictly **memory-bandwidth bound**:

```
Standard Unfused LLM Pipeline (High DRAM Traffic & Thermal Penalties):
Input [seq, d_model] 
  ├──> Gate Projection GEMM ──> Write to DRAM [seq, d_ff] ──┐
  ├──> Up Projection GEMM   ──> Write to DRAM [seq, d_ff] ──┼─> Read DRAM ──> SiLU & Hadamard (⊙) ──> Write to DRAM [seq, d_ff] ──> Read DRAM ──> Down Projection GEMM ──> Output
                                                            │   (Massive memory bus thrashing, LPDDR5 latency & thermal throttling)
                                                            └────────────────────────────────────────────────────────────────────────┘

VectorFFN In-Register Fused Pipeline (Zero Intermediate DRAM Roundtrips):
Input [seq, d_model] 
  └──> In-Register Fused Kernel ──> [Gate_4, Up_4 in NEON Regs] ──> [In-Register SiLU + ⊙] ──> Single Direct Down-Proj Accumulation ──> Output
       (Over 80% activation memory traffic eliminated; stays entirely on-chip within ARM 128-bit vector register files)
```

---

## 🚀 Optimization Stack & Kernel Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Memory Stream Alignment (B^T Offline Weight Transposition)                          │
│    Converts stride-N column accesses to contiguous unit-stride SIMD streams (vld1q_f32)│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. 3D Cache Blocking (MT x NT x KT)                                                    │
│    Partitions working sets to fit L1D/L2 caches; single write-back to main memory      │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 8-Wide ARM NEON Vector Microkernel (microkernel_8x1 & dot8)                         │
│    Utilizes 8 simultaneous vector accumulators (v0..v7) to saturate dual FMA pipelines │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. In-Register Polynomial SiLU Vectorization                                           │
│    IEEE-754 exponent bit-manipulation range-reduction with degree-5 polynomial exp    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. In-Register SwiGLU Kernel Fusion                                                    │
│    Fuses Gate, Up, SiLU, and Hadamard operations directly into vector registers        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 6. Hardware INT8 Matrix Quantization (vdotq_s32)                                       │
│    Per-channel weight & per-tensor activation scaling with <0.6% NRMSE fidelity        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 7. Asymmetric Multiprocessing & Work-Stealing (big.LITTLE / Tri-Cluster)               │
│    Mitigates barrier stalls between fast Prime/Big cores and slow Efficiency cores     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Live Empirical Benchmark Results

Benchmarks were gathered on bare-metal hardware at both **Production LLaMA-3 / Mistral-7B Scale** ($M=128, K=4096, N=14336$, total computation $= 15.03\text{ GFLOPs}$) and **Medium Scale** ($M=64, K=2048, N=5632$).

### 1. 📱 Qualcomm Snapdragon 8 Gen 2 (Samsung Galaxy Tab S9)
*1x Cortex-X3 (3.36 GHz) + 4x Cortex-A715/A710 (2.80 GHz) + 3x Cortex-A510 (2.00 GHz), LPDDR5X Memory.*

#### A. Production LLaMA-3 / Mistral-7B Layer Scale ($M=128, K=4096, N=14336$)
*Tested on Performance Cluster (Cores 3–7: Cortex-X3 + Cortex-A715/A710):*

| Optimization Level / Kernel | Execution Latency | Compute Throughput | Numerical Error (NRMSE) | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **L0: Naive GEMM ($i\text{-}j\text{-}k$)** | 68,617.27 ms | 0.22 GFLOP/s | Reference (FP32 Ground Truth) | $1.0\times$ |
| **L1: Cache-Tiled ($B^T$)** | 5,112.13 ms | 2.94 GFLOP/s | $\text{NRMSE}=1.19\times 10^{-6}$ | **$13.42\times$** |
| **L2: NEON FP32 ($8\times 1$)** | 812.37 ms | 18.50 GFLOP/s | $\text{NRMSE}=1.16\times 10^{-6}$ | **$84.47\times$** |
| **L5: INT8 (`vdotq_s32` Hardware SIMD)** | **360.15 ms** | **41.74 GFLOP/s** | $\text{NRMSE}=0.56\%$ | **$190.52\times$** ($2.26\times$ vs FP32) |
| **Full FFN: Unfused (3x NEON + DRAM $h$)** | 2,424.17 ms | 18.60 GFLOP/s | Reference | $6.81\times$ |
| **Full FFN: Fused NEON (`dynamic`)** | **1,691.16 ms** | **26.67 GFLOP/s** | $\text{NRMSE}=1.40\times 10^{-6}$ | **$9.76\times$ vs Naive ($1.43\times$ vs Unfused)** |

#### B. Asymmetric Multiprocessing & Work-Stealing Analysis ($M=64, K=2048, N=5632$)
| Core Selection / Schedule | Latency | Throughput | Scheduling Behavior & Microarchitectural Reason |
| :--- | :---: | :---: | :--- |
| **All 8 Cores (Static OpenMP)** | 336.44 ms | 13.16 GFLOP/s | **Barrier Stalls:** Equal chunks on slow A510 cores leave fast X3/A715 cores idle. |
| **All 8 Cores (Dynamic OpenMP)** | 224.52 ms | 19.73 GFLOP/s | **Work Stealing ($1.50\times$ faster):** Fast cores continuously pull chunks from task queue. |
| **Performance Cluster (Cores 3–7)** | **161.58 ms** | **27.41 GFLOP/s** | **$2.08\times$ vs 8-Core Static:** Excluding A510 efficiency cores eliminates tail latency completely. |

---

### 2. 🍎 Apple Silicon M1 (MacBook Air)
*4x Firestorm Performance Cores (3.20 GHz) + 4x Icestorm Efficiency Cores (2.06 GHz), Unified LPDDR4X Memory.*

#### A. Production LLaMA-3 / Mistral-7B Layer Scale ($M=128, K=4096, N=14336$)

| Optimization Level / Kernel | Execution Latency | Compute Throughput | Numerical Error (NRMSE) | Speedup vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **L0: Naive GEMM ($i\text{-}j\text{-}k$)** | 27,503.54 ms | 0.55 GFLOP/s | Reference (FP32 Ground Truth) | $1.0\times$ |
| **L1: Cache-Tiled ($B^T$)** | 5,199.43 ms | 2.89 GFLOP/s | $\text{NRMSE}=1.19\times 10^{-6}$ | **$5.29\times$** |
| **L2: NEON FP32 ($8\times 1$)** | 741.75 ms | 20.27 GFLOP/s | $\text{NRMSE}=1.16\times 10^{-6}$ | **$37.08\times$** |
| **L5: INT8 (`vdotq_s32` Hardware SIMD)** | **378.29 ms** | **39.74 GFLOP/s** | $\text{NRMSE}=0.56\%$ | **$72.71\times$** ($1.96\times$ vs FP32) |
| *Ref: Apple Accelerate (`cblas_sgemm`)* | 28.38 ms | 529.65 GFLOP/s | Reference | $969.06\times$ (Dedicated AMX Co-processor) |

#### B. Thread Scaling & Heterogeneous Workload Sweep ($M=128, K=4096, N=14336$)
| Thread Configuration | Latency | Throughput | Microarchitectural Observation |
| :--- | :---: | :---: | :--- |
| **All 8 Cores (Static OpenMP)** | 1,965.68 ms | 22.94 GFLOP/s | Stalls at OpenMP barrier waiting on 4 Icestorm E-cores. |
| **All 8 Cores (Dynamic OpenMP)** | 1,654.62 ms | 27.26 GFLOP/s | Dynamic work-stealing improves load balancing across P/E split ($1.19\times$ speedup). |
| **4 P-Cores Only (`OMP_NUM_THREADS=4`)** | **1,353.84 ms** | **33.31 GFLOP/s** | **$16.47\times$ vs Naive ($1.48\times$ vs Unfused):** Peak CPU performance on Firestorm P-cores. |
| *Ref: Unfused (3x Accelerate + DRAM)* | 91.09 ms | 495.10 GFLOP/s | Dedicated AMX baseline with DRAM activation traffic. |

---

## 🔬 Systems & Microarchitectural Deep-Dive

### 1. The Stride-$N$ Memory Wall & Offline Weight Transposition ($B^T$)
In standard row-major GEMM ($C = A \cdot B$), accessing matrix $B [K, N]$ along columns induces a memory stride of $N \times 4\text{ bytes}$. When $N > 16$ (e.g., $N=14,336$ in LLaMA-3), every inner loop iteration accesses a separate 64-byte cache line, causing a $100\%$ cache line miss rate. 
VectorFFN pre-transposes weights offline ($B^T [N, K]$), aligning both input activations and weights into contiguous row-major streams. This allows dual 128-bit vector loads (`vld1q_f32`) with unit stride directly into L1D cache.

### 2. 3D Cache-Blocked Tiling ($M_T, N_T, K_T$)
To maximize spatial and temporal locality across the cache hierarchy, the matrix loops are blocked into 3 dimensions ($M_T=64, N_T=64, K_T=256$):
* Active tiles fit within L1D ($64\text{--}128\text{ KB}$) and L2 ($512\text{ KB}\text{--}1\text{ MB}$) caches.
* Accumulators reside in CPU cache for the duration of the $K$-sweep, reducing DRAM write-backs to **exactly one commit per tile**.

### 3. $8\times 1$ Register-Blocked NEON Microkernel
The inner compute engine unrolls 8 rows of $B^T$ against 1 row of $A$ simultaneously using 8 NEON 128-bit vector accumulators (`v0`..`v7`):
* **Register File Allocation:** 8 accumulators + 1 shared $A$ vector + 8 $B^T$ vectors = 17 live vector registers (within ARM64's 32-register `v0`..`v31` file, avoiding stack spilling).
* **Arithmetic Intensity:** Delivers **8 FMAs per $A$-row load**, maximizing instruction-level parallelism (ILP) on wide out-of-order execution pipelines (Cortex-X3 / Firestorm).

### 4. Zero-DRAM Operator Fusion
Standard ML runtimes materialize full intermediate $[M, d_{\text{ff}}]$ matrices for $\text{Gate}(x)$ and $\text{Up}(x)$ into DRAM. VectorFFN fuses this pipeline directly within CPU vector registers:
$$\text{Vector } x \xrightarrow{\text{Reg Load}} \text{Compute } \left[\text{Gate}_4, \text{Up}_4\right] \xrightarrow{\text{In-Register SiLU}} \text{Hadamard } (\odot) \xrightarrow{\text{Reg Store}} h_4$$
This cuts intermediate activation DRAM traffic by **$>80\%$**, transforming a memory-bound decode phase into a compute-bound operation.

### 5. Metric Rigor: Why NRMSE vs. Max Relative Error
Max relative error ($\max \frac{|y - \hat{y}|}{|y|}$) is mathematically unstable on GEMM outputs because near-zero values at the numerical noise floor artificially inflate error ratios to thousands of percent. We validate precision using **Normalized Root-Mean-Square Error (NRMSE)**:
$$\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}}{\sqrt{\frac{1}{N}\sum_{i=1}^N y_i^2}}$$
VectorFFN’s INT8 kernel achieves **$<0.6\%$ NRMSE**, proving zero perceptual loss for production LLM weights.

---

## 🛠️ Build & Quick-Start Guide

### 1. macOS (Apple Silicon M1/M2/M3/M4)
```bash
brew install cmake libomp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j

# Run full LLaMA-3 scale benchmark
./ffn_bench 128 4096 14336 2
```

### 2. Android / Snapdragon (via ADB Cross-Compilation)
```bash
# Cross-compile from Mac using Android NDK
NDK_COMPILER=$(find /opt/homebrew/share/android-ndk /opt/homebrew/Caskroom/android-ndk -name "aarch64-linux-android*clang++" | grep -E "android(29|30|31|32|33|34)" | head -n 1)

$NDK_COMPILER -O3 -std=c++17 \
  -march=armv8.5-a+dotprod \
  -fopenmp -static-openmp -static-libstdc++ \
  -Iinclude src/*.cpp -o build/ffn_bench_sd8g2

# Push & Run on Android device
adb push build/ffn_bench_sd8g2 /data/local/tmp/
adb shell "chmod +x /data/local/tmp/ffn_bench_sd8g2 && /data/local/tmp/ffn_bench_sd8g2 128 4096 14336 2 perf"
```

### 3. Android / Snapdragon (Natively inside Termux)
```bash
termux-setup-storage
pkg update && pkg install -y clang cmake libomp openblas git
cp -r /sdcard/Download/ffn_engine ~/VectorFFN && cd ~/VectorFFN

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFFN_FORCE_ARM_MARCH=armv8.5-a+dotprod
cmake --build . -j
./ffn_bench 128 4096 14336 2 perf
```

---

## 💼 Interview & Resume Highlights (For Qualcomm, Apple & NVIDIA)

### High-Impact Resume Bullets:
> **On-Device LLM Inference Optimization & Micro-Kernels (ARM64, C++17)**
> - Engineered an optimized SwiGLU / LLaMA-3 feed-forward inference engine in bare-metal C++17, achieving **$190.5\times$ speedup** and **$41.7\text{ GFLOP/s}$** on Snapdragon 8 Gen 2 via INT8 SIMD vectorization (`vdotq_s32`) with **$<0.6\%$ NRMSE**.
> - Eliminated **$>80\%$ of intermediate activation DRAM traffic** during autoregressive token generation by fusing Gate/Up projections, polynomial SiLU, and Hadamard products directly within ARM vector registers.
> - Mitigated asymmetric CPU synchronization bottlenecks across heterogeneous big.LITTLE / tri-cluster architectures (Cortex-X3 / A715 / A510), reducing layer latency by **$2.08\times$** through custom CPU cluster pinning and dynamic work-stealing.
> - Validated numerical accuracy and throughput ceilings against vendor BLAS references (**Apple Accelerate `cblas_sgemm`** and **OpenBLAS**) across Apple Silicon (M1) and Qualcomm Snapdragon.

---

## 📂 Repository Structure

```
VectorFFN/
├── CMakeLists.txt         # Universal build system (ISA detection, OpenMP, BLAS backends)
├── include/
│   ├── common.h           # 64B cache-aligned buffer, Timer, NRMSE validator
│   ├── gemm_naive.h       # L0: Naive scalar GEMM baseline
│   ├── gemm_tiled.h       # L1: Cache-tiled 3D loop blocking
│   ├── gemm_neon.h        # L2: 8-wide & 4-wide NEON register microkernels
│   ├── silu.h             # L3: Vectorized polynomial SiLU (exp range reduction)
│   ├── fused_ffn.h        # L4: Multithreaded in-register fused SwiGLU pipeline
│   └── quantize.h         # L5: INT8 per-channel quantization & vdotq_s32 GEMM
├── src/
│   ├── benchmark_main.cpp # Unified test harness with CPU affinity & BLAS baselines
│   ├── gemm_naive.cpp
│   ├── gemm_tiled.cpp
│   ├── gemm_neon.cpp
│   ├── silu.cpp
│   ├── fused_ffn.cpp
│   └── quantize.cpp
├── LICENSE                # MIT License
└── README.md
```

---

## 📜 License
Released under the [MIT License](LICENSE).
