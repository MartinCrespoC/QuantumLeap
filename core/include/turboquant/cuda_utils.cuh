#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <memory>

namespace turboquant {
namespace cuda {

// ============================================
// Error Checking
// ============================================

#define CUDA_CHECK(call)                                                      \
  do {                                                                        \
    cudaError_t err = (call);                                                 \
    if (err != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
              cudaGetErrorString(err));                                        \
      abort();                                                                \
    }                                                                         \
  } while (0)

#define CUBLAS_CHECK(call)                                                    \
  do {                                                                        \
    cublasStatus_t err = (call);                                              \
    if (err != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,     \
              static_cast<int>(err));                                          \
      abort();                                                                \
    }                                                                         \
  } while (0)

// ============================================
// RAII GPU Memory
// ============================================

struct CudaDeleter {
  void operator()(void* ptr) const {
    if (ptr) cudaFree(ptr);
  }
};

template <typename T>
using CudaUniquePtr = std::unique_ptr<T, CudaDeleter>;

template <typename T>
CudaUniquePtr<T> cuda_alloc(size_t count) {
  T* ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
  return CudaUniquePtr<T>(ptr);
}

// ============================================
// Memory Transfer
// ============================================

template <typename T>
void cuda_copy_h2d(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T>
void cuda_copy_d2h(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost));
}

template <typename T>
void cuda_copy_d2d(T* dst, const T* src, size_t count) {
  CUDA_CHECK(cudaMemcpy(dst, src, count * sizeof(T), cudaMemcpyDeviceToDevice));
}

// ============================================
// Kernel Launch Utilities
// ============================================

// Tile sizes for matrix operations
constexpr int TILE_SIZE = 32;
constexpr int WARP_SIZE = 32;
constexpr int MAX_THREADS_PER_BLOCK = 1024;

// Calculate grid dimensions
inline dim3 grid_dim(int total, int block_size) {
  return dim3((total + block_size - 1) / block_size);
}

inline dim3 grid_dim_2d(int rows, int cols, int block_x, int block_y) {
  return dim3((cols + block_x - 1) / block_x, (rows + block_y - 1) / block_y);
}

// Get optimal block size for a kernel
template <typename KernelFunc>
int optimal_block_size(KernelFunc kernel) {
  int min_grid_size, block_size;
  cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, kernel);
  return block_size;
}

// ============================================
// Device Info
// ============================================

inline int device_count() {
  int count = 0;
  CUDA_CHECK(cudaGetDeviceCount(&count));
  return count;
}

inline size_t free_memory(int device = 0) {
  CUDA_CHECK(cudaSetDevice(device));
  size_t free_mem, total_mem;
  CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
  return free_mem;
}

inline int compute_capability(int device = 0) {
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  return prop.major * 10 + prop.minor;
}

}  // namespace cuda
}  // namespace turboquant
