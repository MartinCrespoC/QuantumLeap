---
description: How to optimize vector operations with AVX-512 SIMD intrinsics
---

# Skill: Optimize with SIMD/AVX-512

## When to use
- Vector operations on arrays >100 elements
- Hot loops identified by `perf record`
- Data-parallel computations (quantize, dequantize, dot product)

## Steps
1. Identify loop to vectorize with `perf report`
2. Ensure data alignment: `alignas(64)` or `aligned_alloc(64, size)`
3. Use intrinsics: `_mm512_*` for AVX-512 (16 floats at a time)
4. Handle remainder elements with scalar fallback
5. Verify with `perf stat -e cycles,instructions`

## Key intrinsics
```cpp
__m512 v = _mm512_load_ps(ptr);           // Load 16 aligned floats
_mm512_store_ps(ptr, v);                   // Store 16 aligned floats
__m512 r = _mm512_fmadd_ps(a, b, c);      // a*b + c (fused)
float s = _mm512_reduce_add_ps(v);         // Horizontal sum
__m512i q = _mm512_cvtps_epi32(v);         // Float → Int32
```

## Verification
- Expected speedup: 8-16x for simple ops vs scalar
- Check vectorization: `objdump -d -M intel | grep zmm`
- Profile: `perf stat -e fp_arith_inst_retired.512b_packed_single`
