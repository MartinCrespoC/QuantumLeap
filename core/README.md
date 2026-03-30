# TurboQuant Core (C++/CUDA/Assembly)

High-performance quantization kernels for LLM inference.

## Components

### `src/turboquant/`
- **`polar_transform.cpp`** — PolarQuant CPU implementation with runtime dispatch
- **`polar_transform_avx512.S`** — AVX-512 assembly (~4x faster than scalar)
- **`residual_quant.cpp`** — Iterative residual quantization (INT2/INT4)
- **`lookup_tables.cpp`** — Pre-computed lookup tables for fast dequant
- **`kv_cache_compress.cu`** — CUDA KV cache compression

### `src/kernels/`
- **`matmul_int2.cu`** — INT2 matrix multiply (shared memory tiled)
- **`matmul_int4.cu`** — INT4 matrix multiply
- **`attention_compressed.cu`** — Attention with compressed KV cache
- **`dequant_fast.S`** — AVX-512 fast dequantization (~1 cycle/element)

### `src/llama_integration/`
- **`ggml_turboquant.cpp`** — GGML backend for TurboQuant format
- **`llama_cpp_patch.cpp`** — llama.cpp integration and auto-offloading

## Build

```bash
cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DAVX512=ON -DCUDA_ARCH=86
cmake --build build -j12
```

## Test

```bash
cd build && ctest --output-on-failure
./test_all      # Accuracy tests
./benchmark     # Performance benchmarks
```
