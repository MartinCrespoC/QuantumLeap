#include "turboquant/turboquant.h"
#include "turboquant/simd_utils.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace turboquant {

namespace {

// Pre-computed lookup tables for fast dequantization
// INT2: 4 possible values × group_size combinations
// INT4: 16 possible values × group_size combinations

alignas(64) float g_int2_lut[4];      // INT2 centered values
alignas(64) float g_int4_lut[16];     // INT4 centered values
alignas(64) float g_atan2_lut[256];   // Fast atan2 lookup (8-bit angle)

bool g_tables_initialized = false;

}  // namespace

void init_lookup_tables() {
  if (g_tables_initialized) return;

  // INT2 lookup: values 0,1,2,3 centered at 1.5
  for (int i = 0; i < 4; ++i) {
    g_int2_lut[i] = static_cast<float>(i) - 1.5f;
  }

  // INT4 lookup: values 0-15 centered at 7.5
  for (int i = 0; i < 16; ++i) {
    g_int4_lut[i] = static_cast<float>(i) - 7.5f;
  }

  // Fast atan2 lookup table (8-bit precision)
  // Maps angle index [0, 255] → [-pi, pi]
  for (int i = 0; i < 256; ++i) {
    g_atan2_lut[i] = (static_cast<float>(i) / 255.0f) * 2.0f *
                     static_cast<float>(M_PI) - static_cast<float>(M_PI);
  }

  g_tables_initialized = true;
}

void destroy_lookup_tables() {
  g_tables_initialized = false;
}

}  // namespace turboquant
