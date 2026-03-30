---
description: Checklist and techniques for optimizing CUDA kernels on RTX 3050
---

# Skill: Optimize CUDA Kernels

## Target: RTX 3050 Laptop (Compute 8.6, 2048 cores, 4GB VRAM)

## Checklist
- [ ] Coalesced global memory access (128-byte aligned)
- [ ] Shared memory for tile-based algorithms
- [ ] Occupancy >70% (`nvprof --metrics achieved_occupancy`)
- [ ] No bank conflicts in shared memory
- [ ] Register pressure <64 per thread
- [ ] Warp divergence minimized
- [ ] Use `__restrict__` on all pointer params
- [ ] Use `--use_fast_math` for approx math

## Key patterns
1. **Tiling**: Load tiles into shared memory, compute locally
2. **Warp primitives**: `__shfl_*` for intra-warp reductions
3. **Async copy**: `memcpy_async` for pipelined loads
4. **FP16 math**: Use `half` type for 2x throughput

## Profiling commands
```bash
nvprof --metrics all ./kernel_benchmark
nvprof --print-gpu-trace ./kernel_benchmark
ncu --set full ./kernel_benchmark  # Nsight Compute
```

## Memory bandwidth target
- RTX 3050 Laptop: ~192 GB/s theoretical
- Good kernel: >120 GB/s achieved (>60% utilization)
