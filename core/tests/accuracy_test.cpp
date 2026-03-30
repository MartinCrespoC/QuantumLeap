#include "turboquant/turboquant.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace turboquant;

static void fill_random(std::vector<float>& v, float lo, float hi) {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(lo, hi);
  for (auto& x : v) x = dist(rng);
}

// Test: PolarQuant round-trip accuracy
static bool test_polar_roundtrip() {
  const size_t n = 1024;
  std::vector<float> x(n), y(n), mag(n), ang(n);
  fill_random(x, -5.0f, 5.0f);
  fill_random(y, -5.0f, 5.0f);

  polar_transform(x.data(), y.data(), mag.data(), ang.data(), n);

  // Reconstruct: x' = mag * cos(ang), y' = mag * sin(ang)
  float max_err = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float x_recon = mag[i] * std::cos(ang[i]);
    float y_recon = mag[i] * std::sin(ang[i]);
    max_err = std::max(max_err, std::abs(x[i] - x_recon));
    max_err = std::max(max_err, std::abs(y[i] - y_recon));
  }

  printf("[PolarQuant Round-trip] max error: %.2e ", max_err);
  bool pass = max_err < 1e-3f;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass;
}

// Test: Residual quantization accuracy (INT2)
static bool test_residual_quant_int2() {
  const size_t n = 4096;
  std::vector<float> data(n);
  fill_random(data, -1.0f, 1.0f);

  auto result = residual_quantize(data.data(), n, QuantBits::kInt2, 128, 3);

  printf("[ResidualQuant INT2] MSE: %.6f, Max Error: %.6f ", result.mse, result.max_error);
  // INT2 with 4 levels: expect reasonable error
  bool pass = result.mse < 0.1f;
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] result.meta.scales;
  delete[] result.meta.zero_points;
  return pass;
}

// Test: Residual quantization accuracy (INT4)
static bool test_residual_quant_int4() {
  const size_t n = 4096;
  std::vector<float> data(n);
  fill_random(data, -1.0f, 1.0f);

  auto result = residual_quantize(data.data(), n, QuantBits::kInt4, 128, 3);

  printf("[ResidualQuant INT4] MSE: %.6f, Max Error: %.6f ", result.mse, result.max_error);
  // INT4 with 16 levels: expect lower error than INT2
  bool pass = result.mse < 0.01f;
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] result.meta.scales;
  delete[] result.meta.zero_points;
  return pass;
}

// Test: Full TurboQuant pipeline
static bool test_turboquant_encode() {
  const size_t n = 2048;  // Must be even for polar transform
  std::vector<float> data(n);
  fill_random(data, -2.0f, 2.0f);

  auto result = turboquant_encode(data.data(), n, QuantBits::kInt2, 128);

  printf("[TurboQuant Encode] MSE: %.6f, Max Error: %.6f, Packed: %zu bytes ",
         result.mse, result.max_error, result.data.size());
  printf("(%.2f bits/elem) ", result.data.size() * 8.0 / n);

  bool pass = result.data.size() > 0 && result.mse < 0.5f;
  printf("%s\n", pass ? "PASS" : "FAIL");

  delete[] result.meta.scales;
  delete[] result.meta.zero_points;
  if (result.meta.magnitudes) delete[] result.meta.magnitudes;
  if (result.meta.angles) delete[] result.meta.angles;
  return pass;
}

// Test: CPU feature detection
static bool test_cpu_features() {
  bool avx512 = has_avx512();
  bool avx2 = has_avx2();

  printf("[CPU Features] AVX-512: %s, AVX2: %s ",
         avx512 ? "YES" : "NO", avx2 ? "YES" : "NO");
  // At least AVX2 should be present on i5-11400H
  bool pass = avx2;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass;
}

int main() {
  printf("=== TurboQuant Accuracy Tests ===\n\n");

  init_lookup_tables();

  int passed = 0, total = 0;

  auto run = [&](bool (*test)()) {
    total++;
    if (test()) passed++;
  };

  run(test_cpu_features);
  run(test_polar_roundtrip);
  run(test_residual_quant_int2);
  run(test_residual_quant_int4);
  run(test_turboquant_encode);

  destroy_lookup_tables();

  printf("\n=== Results: %d/%d passed ===\n", passed, total);
  return (passed == total) ? 0 : 1;
}
