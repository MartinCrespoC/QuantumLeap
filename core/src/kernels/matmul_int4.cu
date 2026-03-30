#include "turboquant/cuda_utils.cuh"

namespace turboquant {

// ============================================
// INT4 Matrix Multiply CUDA Kernel
// A (M x K) @ B (K x N) = C (M x N)
// A and B packed INT4: 2 values per byte
// ============================================

__device__ __forceinline__ float unpack_int4_lo(uint8_t packed) {
  return static_cast<float>(packed & 0xF) - 7.5f;
}

__device__ __forceinline__ float unpack_int4_hi(uint8_t packed) {
  return static_cast<float>((packed >> 4) & 0xF) - 7.5f;
}

__global__ void matmul_int4_kernel(
    const uint8_t* __restrict__ A,
    const uint8_t* __restrict__ B,
    float* __restrict__ C,
    const float* __restrict__ sa,
    const float* __restrict__ sb,
    const int M, const int N, const int K,
    const int group_size) {

  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= M || col >= N) return;

  float acc = 0.0f;

  for (int k = 0; k < K; k += 2) {
    int a_byte = row * (K / 2) + k / 2;
    int b_byte = k * (N / 2) + col / 2;

    float a0 = unpack_int4_lo(A[a_byte]);
    float a1 = unpack_int4_hi(A[a_byte]);

    // B access for two consecutive K values
    float b0, b1;
    if (col % 2 == 0) {
      b0 = unpack_int4_lo(B[k * (N / 2) + col / 2]);
      b1 = unpack_int4_lo(B[(k + 1) * (N / 2) + col / 2]);
    } else {
      b0 = unpack_int4_hi(B[k * (N / 2) + col / 2]);
      b1 = unpack_int4_hi(B[(k + 1) * (N / 2) + col / 2]);
    }

    acc += a0 * b0 + a1 * b1;
  }

  float scale_a = sa[row / group_size];
  float scale_b = sb[col / group_size];
  C[row * N + col] = acc * scale_a * scale_b;
}

void matmul_int4_cuda(
    const uint8_t* A, const uint8_t* B, float* C,
    const float* scales_a, const float* scales_b,
    int M, int N, int K) {

  dim3 block(16, 16);
  dim3 grid(
    (N + block.x - 1) / block.x,
    (M + block.y - 1) / block.y
  );

  matmul_int4_kernel<<<grid, block>>>(
    A, B, C, scales_a, scales_b, M, N, K, 128
  );

  CUDA_CHECK(cudaGetLastError());
}

}  // namespace turboquant
