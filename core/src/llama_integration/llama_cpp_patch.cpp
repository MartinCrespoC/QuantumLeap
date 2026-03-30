#include "turboquant/turboquant.h"

#include <cstdio>

namespace turboquant {

// ============================================
// llama.cpp Integration Patch
// Hooks into llama.cpp model loading and inference
// to use TurboQuant quantized tensors
// ============================================

// Model layer offloading configuration
struct OffloadConfig {
  int total_layers;
  int gpu_layers;         // Layers fully on GPU
  int cpu_layers;         // Layers on CPU (RAM)
  size_t vram_budget;     // Max VRAM to use (bytes)
  size_t ram_budget;      // Max RAM to use (bytes)
  QuantBits weight_bits;  // Quantization for weights
  QuantBits kv_bits;      // Quantization for KV cache
};

// Auto-calculate optimal layer distribution
OffloadConfig auto_offload_config(
    int total_layers, size_t param_count, size_t vram_available) {

  OffloadConfig config;
  config.total_layers = total_layers;
  config.vram_budget = vram_available;

  // Estimate bytes per layer based on quantization
  // INT2: ~0.25 bytes/param, INT4: ~0.5 bytes/param
  size_t bytes_per_layer_tq2 = (param_count / total_layers) / 4;
  size_t bytes_per_layer_tq4 = (param_count / total_layers) / 2;

  // Reserve 512MB for KV cache and runtime overhead
  size_t usable_vram = (vram_available > 512 * 1024 * 1024)
                           ? vram_available - 512 * 1024 * 1024
                           : 0;

  // Try INT2 first (more layers fit in VRAM)
  int gpu_layers_tq2 = static_cast<int>(usable_vram / bytes_per_layer_tq2);
  if (gpu_layers_tq2 > total_layers) gpu_layers_tq2 = total_layers;

  int gpu_layers_tq4 = static_cast<int>(usable_vram / bytes_per_layer_tq4);
  if (gpu_layers_tq4 > total_layers) gpu_layers_tq4 = total_layers;

  // Use INT2 if it gets significantly more layers on GPU
  if (gpu_layers_tq2 > gpu_layers_tq4 * 1.3) {
    config.gpu_layers = gpu_layers_tq2;
    config.weight_bits = QuantBits::kInt2;
  } else {
    config.gpu_layers = gpu_layers_tq4;
    config.weight_bits = QuantBits::kInt4;
  }

  config.cpu_layers = total_layers - config.gpu_layers;
  config.kv_bits = QuantBits::kInt2;  // Always use INT2 for KV cache
  config.ram_budget = 0;  // Will be calculated at runtime

  printf("[TurboQuant] Auto-offload config:\n");
  printf("  Total layers: %d\n", config.total_layers);
  printf("  GPU layers: %d (%.0f%%)\n", config.gpu_layers,
         100.0f * config.gpu_layers / total_layers);
  printf("  CPU layers: %d\n", config.cpu_layers);
  printf("  Weight quant: INT%d\n", static_cast<int>(config.weight_bits));
  printf("  KV cache quant: INT%d\n", static_cast<int>(config.kv_bits));
  printf("  VRAM budget: %.1f MB\n", vram_available / (1024.0 * 1024.0));

  return config;
}

// Print model size comparison
void print_size_comparison(size_t param_count) {
  printf("\n[TurboQuant] Model size comparison (%zu params):\n", param_count);
  printf("  FP16:     %.1f GB\n", param_count * 2.0 / (1024 * 1024 * 1024));
  printf("  Q8_0:     %.1f GB\n", param_count * 1.0 / (1024 * 1024 * 1024));
  printf("  Q4_K_M:   %.1f GB\n", param_count * 0.5 / (1024 * 1024 * 1024));
  printf("  TQ4:      %.1f GB\n", param_count * 0.516 / (1024 * 1024 * 1024));
  printf("  TQ2:      %.1f GB  ← TurboQuant\n",
         param_count * 0.266 / (1024 * 1024 * 1024));
  printf("  Savings vs Q4_K_M: %.0f%%\n",
         (1.0 - 0.266 / 0.5) * 100.0);
}

}  // namespace turboquant
