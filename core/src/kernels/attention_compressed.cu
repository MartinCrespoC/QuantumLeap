#include "turboquant/cuda_utils.cuh"
#include "turboquant/turboquant.h"

namespace turboquant {

// ============================================
// Compressed Attention CUDA Kernel
// Q @ K_compressed^T * V_compressed
// KV cache stored in INT2/INT4 to save VRAM
// ============================================

#define HEAD_DIM_MAX 128
#define WARP_SIZE 32

// Decompress and compute attention score for one query against compressed keys
__global__ void attention_score_compressed_kernel(
    const float* __restrict__ Q,           // [batch, heads, 1, head_dim]
    const uint8_t* __restrict__ K_comp,    // [batch, heads, seq_len, head_dim/pack]
    const float* __restrict__ K_scales,    // [batch, heads, seq_len, groups]
    float* __restrict__ scores,            // [batch, heads, seq_len]
    const int batch, const int heads,
    const int seq_len, const int head_dim,
    const int bits, const int group_size) {

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int s = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= batch || h >= heads || s >= seq_len) return;

  int pack_ratio = 8 / bits;
  int mask = (1 << bits) - 1;
  float inv_sqrt_d = rsqrtf(static_cast<float>(head_dim));

  float dot = 0.0f;

  int q_offset = (b * heads + h) * head_dim;
  int k_offset = ((b * heads + h) * seq_len + s) * (head_dim / pack_ratio);
  int scale_offset = ((b * heads + h) * seq_len + s) * (head_dim / group_size);

  for (int d = 0; d < head_dim; ++d) {
    float q_val = Q[q_offset + d];

    // Decompress K value
    int byte_idx = k_offset + d / pack_ratio;
    int sub_idx = d % pack_ratio;
    uint8_t packed = K_comp[byte_idx];
    int q_int = (packed >> (sub_idx * bits)) & mask;

    float scale = K_scales[scale_offset + d / group_size];
    float k_val = q_int * scale;

    dot += q_val * k_val;
  }

  scores[(b * heads + h) * seq_len + s] = dot * inv_sqrt_d;
}

// Apply softmax and multiply by compressed V
__global__ void attention_output_compressed_kernel(
    const float* __restrict__ attn_weights,  // [batch, heads, seq_len] (softmaxed)
    const uint8_t* __restrict__ V_comp,      // [batch, heads, seq_len, head_dim/pack]
    const float* __restrict__ V_scales,      // [batch, heads, seq_len, groups]
    float* __restrict__ output,              // [batch, heads, head_dim]
    const int batch, const int heads,
    const int seq_len, const int head_dim,
    const int bits, const int group_size) {

  const int b = blockIdx.z;
  const int h = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;

  if (b >= batch || h >= heads || d >= head_dim) return;

  int pack_ratio = 8 / bits;
  int mask = (1 << bits) - 1;
  float acc = 0.0f;

  for (int s = 0; s < seq_len; ++s) {
    float weight = attn_weights[(b * heads + h) * seq_len + s];

    // Decompress V value
    int v_offset = ((b * heads + h) * seq_len + s) * (head_dim / pack_ratio);
    int byte_idx = v_offset + d / pack_ratio;
    int sub_idx = d % pack_ratio;
    uint8_t packed = V_comp[byte_idx];
    int v_int = (packed >> (sub_idx * bits)) & mask;

    int scale_offset = ((b * heads + h) * seq_len + s) * (head_dim / group_size);
    float scale = V_scales[scale_offset + d / group_size];
    float v_val = v_int * scale;

    acc += weight * v_val;
  }

  output[(b * heads + h) * head_dim + d] = acc;
}

// Host function
void attention_compressed_cuda(
    const float* Q,
    const uint8_t* K_compressed,
    const uint8_t* V_compressed,
    const float* K_scales,
    const float* V_scales,
    float* output,
    int batch, int heads, int seq_len, int head_dim,
    QuantBits kv_bits) {

  int bits = static_cast<int>(kv_bits);
  int group_size = 128;

  // Step 1: Compute attention scores
  float* d_scores;
  size_t scores_size = batch * heads * seq_len * sizeof(float);
  CUDA_CHECK(cudaMalloc(&d_scores, scores_size));

  {
    dim3 block(256);
    dim3 grid((seq_len + 255) / 256, heads, batch);
    attention_score_compressed_kernel<<<grid, block>>>(
      Q, K_compressed, K_scales, d_scores,
      batch, heads, seq_len, head_dim, bits, group_size
    );
    CUDA_CHECK(cudaGetLastError());
  }

  // TODO: Apply softmax to d_scores (use cuDNN or custom kernel)

  // Step 2: Weighted sum with compressed V
  {
    dim3 block(128);
    dim3 grid((head_dim + 127) / 128, heads, batch);
    attention_output_compressed_kernel<<<grid, block>>>(
      d_scores, V_compressed, V_scales, output,
      batch, heads, seq_len, head_dim, bits, group_size
    );
    CUDA_CHECK(cudaGetLastError());
  }

  CUDA_CHECK(cudaFree(d_scores));
}

}  // namespace turboquant
