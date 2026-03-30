---
description: Build TurboQuant core library and run full test suite
---

# Build and Test Workflow

1. Clean and configure CMake build
```bash
cd ~/Documentos/Proyectos/llm-turbo/core && rm -rf build && cmake -B build -GNinja -DCMAKE_BUILD_TYPE=Release -DAVX512=ON -DCUDA_ARCH=86
```

2. Build with all cores
// turbo
```bash
cd ~/Documentos/Proyectos/llm-turbo/core && cmake --build build -j12
```

3. Run accuracy tests
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build && ./test_all
```

4. Run benchmarks
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build && ./benchmark
```

5. Check for memory leaks (optional)
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build && valgrind --leak-check=full ./test_all
```

6. CUDA profile (optional, if GPU benchmarks exist)
```bash
cd ~/Documentos/Proyectos/llm-turbo/core/build && nvprof --analysis-metrics ./benchmark_cuda
```
