#include "turboquant/turboquant.h"
#include "turboquant/simd_utils.h"

#include <cstdio>
#include <cstring>
#include <vector>

namespace turboquant {

// ============================================
// GGML Backend Integration for TurboQuant
// Provides quantization/dequantization in GGUF format
// Compatible with llama.cpp tensor operations
// ============================================

// TurboQuant GGUF type identifiers (custom extension)
// These extend the existing GGML_TYPE_* enum
constexpr int GGML_TYPE_TQ2 = 100;  // TurboQuant INT2
constexpr int GGML_TYPE_TQ4 = 101;  // TurboQuant INT4

// Block size for TurboQuant quantization (matches GGML block convention)
constexpr int TQ_BLOCK_SIZE = 256;

// ============================================
// TurboQuant Block Layout (INT2)
// ============================================
// Per block of 256 elements:
//   - 2 bytes: scale (FP16)
//   - 2 bytes: zero_point (FP16)
//   - 64 bytes: packed INT2 data (256 values × 2 bits / 8)
//   Total: 68 bytes per block (0.266 bytes/element)
//   vs Q4_K_M: ~146 bytes per 256-element block (0.57 bytes/element)

struct BlockTQ2 {
  uint16_t scale;         // FP16 scale factor
  uint16_t zero_point;    // FP16 zero point
  uint8_t data[64];       // 256 INT2 values packed (4 per byte)
};

// ============================================
// TurboQuant Block Layout (INT4)
// ============================================
// Per block of 256 elements:
//   - 2 bytes: scale (FP16)
//   - 2 bytes: zero_point (FP16)
//   - 128 bytes: packed INT4 data (256 values × 4 bits / 8)
//   Total: 132 bytes per block (0.516 bytes/element)

struct BlockTQ4 {
  uint16_t scale;
  uint16_t zero_point;
  uint8_t data[128];
};

// FP32 ↔ FP16 conversion helpers
static inline uint16_t fp32_to_fp16(float val) {
  // Simple truncation (for production, use hardware conversion)
  union { float f; uint32_t u; } bits = {val};
  uint32_t sign = (bits.u >> 16) & 0x8000;
  int32_t exp = ((bits.u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = (bits.u >> 13) & 0x3FF;

  if (exp <= 0) return static_cast<uint16_t>(sign);
  if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);
  return static_cast<uint16_t>(sign | (exp << 10) | mant);
}

static inline float fp16_to_fp32(uint16_t val) {
  uint32_t sign = (val & 0x8000) << 16;
  int32_t exp = (val >> 10) & 0x1F;
  uint32_t mant = val & 0x3FF;

  if (exp == 0) return 0.0f;
  if (exp == 31) {
    union { uint32_t u; float f; } inf = {sign | 0x7F800000};
    return inf.f;
  }

  exp = exp - 15 + 127;
  union { uint32_t u; float f; } result = {sign | (exp << 23) | (mant << 13)};
  return result.f;
}

// ============================================
// Quantize tensor to TurboQuant format
// ============================================

size_t ggml_turboquant_quantize(
    const float* src, void* dst, size_t n, QuantBits bits) {

  size_t num_blocks = (n + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;
  size_t bytes_written = 0;

  if (bits == QuantBits::kInt2) {
    auto* blocks = static_cast<BlockTQ2*>(dst);

    for (size_t b = 0; b < num_blocks; ++b) {
      size_t start = b * TQ_BLOCK_SIZE;
      size_t end = (start + TQ_BLOCK_SIZE > n) ? n : start + TQ_BLOCK_SIZE;
      size_t count = end - start;

      // Find min/max
      float min_val = src[start], max_val = src[start];
      for (size_t i = start + 1; i < end; ++i) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
      }

      float range = max_val - min_val;
      float scale = (range > 1e-8f) ? range / 3.0f : 1.0f;

      blocks[b].scale = fp32_to_fp16(scale);
      blocks[b].zero_point = fp32_to_fp16(min_val);
      memset(blocks[b].data, 0, sizeof(blocks[b].data));

      // Quantize and pack
      for (size_t i = 0; i < count; ++i) {
        float val = (src[start + i] - min_val) / scale;
        int q = static_cast<int>(val + 0.5f);
        if (q < 0) q = 0;
        if (q > 3) q = 3;
        blocks[b].data[i / 4] |= (q & 0x3) << ((i % 4) * 2);
      }
    }
    bytes_written = num_blocks * sizeof(BlockTQ2);

  } else if (bits == QuantBits::kInt4) {
    auto* blocks = static_cast<BlockTQ4*>(dst);

    for (size_t b = 0; b < num_blocks; ++b) {
      size_t start = b * TQ_BLOCK_SIZE;
      size_t end = (start + TQ_BLOCK_SIZE > n) ? n : start + TQ_BLOCK_SIZE;
      size_t count = end - start;

      float min_val = src[start], max_val = src[start];
      for (size_t i = start + 1; i < end; ++i) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
      }

      float range = max_val - min_val;
      float scale = (range > 1e-8f) ? range / 15.0f : 1.0f;

      blocks[b].scale = fp32_to_fp16(scale);
      blocks[b].zero_point = fp32_to_fp16(min_val);
      memset(blocks[b].data, 0, sizeof(blocks[b].data));

      for (size_t i = 0; i < count; ++i) {
        float val = (src[start + i] - min_val) / scale;
        int q = static_cast<int>(val + 0.5f);
        if (q < 0) q = 0;
        if (q > 15) q = 15;
        if (i % 2 == 0) {
          blocks[b].data[i / 2] = q & 0xF;
        } else {
          blocks[b].data[i / 2] |= (q & 0xF) << 4;
        }
      }
    }
    bytes_written = num_blocks * sizeof(BlockTQ4);
  }

  return bytes_written;
}

// ============================================
// Dequantize TurboQuant tensor back to FP32
// ============================================

void ggml_turboquant_dequantize(
    const void* src, float* dst, size_t n, QuantBits bits) {

  size_t num_blocks = (n + TQ_BLOCK_SIZE - 1) / TQ_BLOCK_SIZE;

  if (bits == QuantBits::kInt2) {
    const auto* blocks = static_cast<const BlockTQ2*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
      float scale = fp16_to_fp32(blocks[b].scale);
      float zero = fp16_to_fp32(blocks[b].zero_point);
      size_t start = b * TQ_BLOCK_SIZE;
      size_t end = (start + TQ_BLOCK_SIZE > n) ? n : start + TQ_BLOCK_SIZE;

      for (size_t i = 0; i < end - start; ++i) {
        int q = (blocks[b].data[i / 4] >> ((i % 4) * 2)) & 0x3;
        dst[start + i] = q * scale + zero;
      }
    }

  } else if (bits == QuantBits::kInt4) {
    const auto* blocks = static_cast<const BlockTQ4*>(src);

    for (size_t b = 0; b < num_blocks; ++b) {
      float scale = fp16_to_fp32(blocks[b].scale);
      float zero = fp16_to_fp32(blocks[b].zero_point);
      size_t start = b * TQ_BLOCK_SIZE;
      size_t end = (start + TQ_BLOCK_SIZE > n) ? n : start + TQ_BLOCK_SIZE;

      for (size_t i = 0; i < end - start; ++i) {
        int q;
        if (i % 2 == 0) {
          q = blocks[b].data[i / 2] & 0xF;
        } else {
          q = (blocks[b].data[i / 2] >> 4) & 0xF;
        }
        dst[start + i] = q * scale + zero;
      }
    }
  }
}

}  // namespace turboquant
