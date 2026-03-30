cmake_minimum_required(VERSION 3.20)
project(TurboQuant LANGUAGES C CXX CUDA ASM)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(AVX512 "Enable AVX-512 optimizations" ON)
option(CUDA_ARCH "Target CUDA architecture" 86)
option(BUILD_TESTS "Build test suite" ON)
option(BUILD_BENCHMARKS "Build benchmarks" ON)
option(ENABLE_ASAN "Enable AddressSanitizer" OFF)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

if(AVX512)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512vbmi2")
  add_compile_definitions(AVX512=1)
endif()

if(ENABLE_ASAN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
  set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")
endif()

# CUDA flags
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math -O3")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ============================================
# TurboQuant Core Library
# ============================================

# Assembly sources (Intel syntax)
set(ASM_SOURCES
  src/turboquant/polar_transform_avx512.S
  src/kernels/dequant_fast.S
)

# C++ sources
set(CPP_SOURCES
  src/turboquant/polar_transform.cpp
  src/turboquant/residual_quant.cpp
  src/turboquant/lookup_tables.cpp
  src/llama_integration/ggml_turboquant.cpp
  src/llama_integration/llama_cpp_patch.cpp
)

# CUDA sources
set(CUDA_SOURCES
  src/turboquant/kv_cache_compress.cu
  src/kernels/matmul_int4.cu
  src/kernels/matmul_int2.cu
  src/kernels/attention_compressed.cu
)

# Main library
add_library(turboquant STATIC ${CPP_SOURCES} ${CUDA_SOURCES} ${ASM_SOURCES})
target_link_libraries(turboquant CUDA::cudart CUDA::cublas)
target_include_directories(turboquant PUBLIC ${CMAKE_SOURCE_DIR}/include)

# ============================================
# Tests
# ============================================
if(BUILD_TESTS)
  enable_testing()

  add_executable(test_all
    tests/benchmark_polar.cpp
    tests/benchmark_matmul.cpp
    tests/accuracy_test.cpp
  )
  target_link_libraries(test_all turboquant)
  add_test(NAME TurboQuantTests COMMAND test_all)
endif()

# ============================================
# Benchmarks
# ============================================
if(BUILD_BENCHMARKS)
  add_executable(benchmark tests/benchmark_polar.cpp tests/benchmark_matmul.cpp)
  target_link_libraries(benchmark turboquant)

  add_executable(benchmark_cuda tests/benchmark_matmul.cpp)
  target_link_libraries(benchmark_cuda turboquant)
endif()

# ============================================
# Install
# ============================================
install(TARGETS turboquant ARCHIVE DESTINATION lib)
install(DIRECTORY include/turboquant DESTINATION include)
