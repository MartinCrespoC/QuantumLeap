---
description: Profile and optimize a CUDA or CPU kernel for maximum performance
---

# Optimize Kernel Workflow

1. Build in profile mode
```bash
cd ~/Documentos/Proyectos/llm-turbo/core && cmake -B build-profile -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo -DAVX512=ON && cmake --build build-profile -j12
```

2. Profile CPU hotspots with perf
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build-profile && perf record -g ./benchmark && perf report --no-children
```

3. Check assembly output of hot functions
```bash
objdump -d -M intel -S ~/Documentos/Proyectos/llm-turbo/core/build-profile/benchmark | less
```

4. For CUDA kernels, profile with nvprof
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build-profile && nvprof --metrics achieved_occupancy,sm_efficiency,dram_read_throughput,dram_write_throughput ./benchmark_cuda
```

5. Verify correctness after optimization
// turbo
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build-profile && ./test_all
```

6. Compare benchmark before/after
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build-profile && ./benchmark 2>&1 | tee /tmp/bench_after.txt && echo "--- Compare with previous ---" && diff /tmp/bench_before.txt /tmp/bench_after.txt || true
```
