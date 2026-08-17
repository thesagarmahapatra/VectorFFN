cmake_minimum_required(VERSION 3.16)
project(VectorFFN CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

add_executable(ffn_bench
  src/gemm_naive.cpp
  src/gemm_tiled.cpp
  src/gemm_neon.cpp
  src/silu.cpp
  src/fused_ffn.cpp
  src/quantize.cpp
  src/benchmark_main.cpp
)
target_include_directories(ffn_bench PRIVATE include)

# ---------------------------------------------------------------------------
# ISA flags. On real ARM64 (M1, Snapdragon via Termux) __ARM_NEON is defined
# automatically by the compiler for aarch64 targets, but the dot-product
# extension (__ARM_FEATURE_DOTPROD, needed for vdotq_s32) requires an
# explicit -march that includes it. armv8.5-a covers M1; +dotprod alone is
# enough and portable to any ARMv8.2+dotprod target (Snapdragon 8 Gen 2's
# cores all support it).
# ---------------------------------------------------------------------------
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  target_compile_options(ffn_bench PRIVATE -march=armv8.5-a+dotprod)
elseif(DEFINED FFN_FORCE_ARM_MARCH)
  # Cross-compile builds (e.g. -DCMAKE_CXX_COMPILER=aarch64-linux-gnu-g++)
  # don't always get CMAKE_SYSTEM_PROCESSOR set correctly; force it.
  target_compile_options(ffn_bench PRIVATE -march=${FFN_FORCE_ARM_MARCH})
else()
  message(STATUS "Non-ARM target detected: building portable scalar fallback (no NEON).")
endif()

# ---------------------------------------------------------------------------
# Vendor BLAS Baseline: Apple Accelerate on macOS, OpenBLAS/BLAS on Linux/Android
# ---------------------------------------------------------------------------
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    target_link_libraries(ffn_bench PRIVATE "${ACCELERATE_FRAMEWORK}")
    target_compile_definitions(ffn_bench PRIVATE FFN_HAVE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1 FFN_HAVE_BLAS=1)
    message(STATUS "Apple Accelerate framework found — cblas_sgemm baseline enabled.")
  endif()
else()
  find_package(BLAS)
  if(BLAS_FOUND)
    target_link_libraries(ffn_bench PRIVATE ${BLAS_LIBRARIES})
    target_compile_definitions(ffn_bench PRIVATE FFN_HAVE_BLAS=1)
    message(STATUS "BLAS found (${BLAS_LIBRARIES}) — cblas_sgemm baseline enabled.")
  else()
    find_library(OPENBLAS_LIB NAMES openblas cblas)
    if(OPENBLAS_LIB)
      target_link_libraries(ffn_bench PRIVATE "${OPENBLAS_LIB}")
      target_compile_definitions(ffn_bench PRIVATE FFN_HAVE_BLAS=1)
      message(STATUS "OpenBLAS found (${OPENBLAS_LIB}) — cblas_sgemm baseline enabled.")
    endif()
  endif()
endif()

# ---------------------------------------------------------------------------
# OpenMP. On macOS/Apple Clang this REQUIRES libomp from Homebrew:
#   brew install libomp
# Apple Clang does not ship libomp, and CMake's FindOpenMP needs the hints
# below to locate it (this is the single most common build failure people
# hit doing exactly this project on an M1 Mac).
# ---------------------------------------------------------------------------
if(APPLE)
  execute_process(COMMAND brew --prefix libomp OUTPUT_VARIABLE LIBOMP_PREFIX OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(LIBOMP_PREFIX)
    set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${LIBOMP_PREFIX}/include")
    set(OpenMP_C_LIB_NAMES "omp")
    set(OpenMP_CXX_LIB_NAMES "omp")
    set(OpenMP_omp_LIBRARY "${LIBOMP_PREFIX}/lib/libomp.dylib")
    set(OpenMP_CXX_INCLUDE_DIRS "${LIBOMP_PREFIX}/include")
  endif()
endif()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(ffn_bench PRIVATE OpenMP::OpenMP_CXX)
  message(STATUS "OpenMP found — multithreaded fused FFN enabled.")
else()
  message(WARNING "OpenMP NOT found — fused_ffn_forward will run single-threaded. "
                   "On macOS: brew install libomp, then re-run cmake.")
endif()

