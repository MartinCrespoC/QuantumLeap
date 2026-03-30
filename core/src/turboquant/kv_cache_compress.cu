#include "turboquant/cuda_utils.cuh"
#include "turboquant/turboquant.h"

namespace turboquant {

// ============================================
// KV Cache Compression CUDA Kernels
// Compresses attention KV cache on-the-fly
// Critical for long context LLM inference
// ============================================

// Quantize FP32 KV cache to INT2/INT4 per group
__global__ void kv_compress_kernel(
    const float* __restrict__ kv_data,   // [seq_len, head_dim]
    uint8_t* __restrict__ compressed,    // [seq_len, head_dim / pack_ratio]
    float* __restrict__ scales,          // [seq_len, head_dim / group_size]
    const int seq_len,
    const int head_dim,
    const int group_size,
    const int bits) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_groups = seq_len * (head_dim / group_size);
  if (tid >= total_groups) return;

  const int seq_idx = tid / (head_dim / group_size);
  const int group_idx = tid % (head_dim / group_size);
  const int offset = seq_idx * head_dim + group_idx * group_size;

  // Find min/max in this group
  float min_val = 1e10f, max_val = -1e10f;
  for (int i = 0; i < group_size; ++i) {
    float v = kv_data[offset + i];
    min_val = fminf(min_val, v);
    max_val = fmaxf(max_val, v);
  }

  int max_quant = (1 << bits) - 1;
  float range = max_val - min_val;
  float scale = (range > 1e-8f) ? range / max_quant : 1.0f;

  // Store scale
  scales[tid] = scale;

  // Quantize and pack
  int pack_ratio = 8 / bits;
  int packed_per_group = group_size / pack_ratio;
  int packed_offset = seq_idx * (head_dim / pack_ratio) +
                      group_idx * packed_per_group;

  for (int i = 0; i < packed_per_group; ++i) {
    uint8_t packed = 0;
    for (int j = 0; j < pack_ratio; ++j) {
      int elem_idx = i * pack_ratio + j;
      float val = kv_data[offset + elem_idx];
      int q = __float2int_rn((val - min_val) / scale);
      q = max(0, min(max_quant, q));
      packed |= (q & ((1 << bits) - 1)) << (j * bits);
    }
    compressed[packed_offset + i] = packed;
  }
}

// Decompress INT2/INT4 KV cache back to FP32
__global__ void kv_decompress_kernel(
    const uint8_t* __restrict__ compressed,
    const float* __restrict__ scales,
    float* __restrict__ output,
    const int seq_len,
    const int head_dim,
    const int group_size,
    const int bits) {

  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_elements = seq_len * head_dim;
  if (tid >= total_elements) return;

  const int seq_idx = tid / head_dim;
  const int dim_idx = tid % head_dim;
  const int group_idx = dim_idx / group_size;
  const int in_group_idx = dim_idx % group_size;

  int pack_ratio = 8 / bits;
  int byte_idx = seq_idx * (head_dim / pack_ratio) + dim_idx / pack_ratio;
  int sub_idx = dim_idx % pack_ratio;

  uint8_t packed = compressed[byte_idx];
  int q = (packed >> (sub_idx * bits)) & ((1 << bits) - 1);

  float scale = scales[seq_idx * (head_dim / group_size) + group_idx];
  output[tid] = q * scale;
}

// Host functions
void kv_cache_compress_cuda(
    const float* kv_data, uint8_t* compressed, float* scales,
    size_t seq_len, size_t head_dim, QuantBits bits) {

  int b = static_cast<int>(bits);
  int group_size = 128;
  int total_groups = seq_len * (head_dim / group_size);

  int threads = 256;
  int blocks = (total_groups + threads - 1) / threads;

  kv_compress_kernel<<<blocks, threads>>>(
    kv_data, compressed, scales,
    static_cast<int>(seq_len), static_cast<int>(head_dim),
    group_size, b
  );
  CUDA_CHECK(cudaGetLastError());
}

void kv_cache_decompress_cuda(
    const uint8_t* compressed, const float* scales, float* output,
    size_t seq_len, size_t head_dim, QuantBits bits) {

  int b = static_cast<int>(bits);
  int group_size = 128;
  int total = seq_len * head_dim;

  int threads = 256;
  int blocks = (total + threads - 1) / threads;

  kv_decompress_kernel<<<blocks, threads>>>(
    compressed, scales, output,
    static_cast<int>(seq_len), static_cast<int>(head_dim),
    group_size, b
  );
  CUDA_CHECK(cudaGetLastError());
}

}  // namespace turboquant
