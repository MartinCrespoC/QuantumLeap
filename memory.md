# QuantumLeap v0.4.0 - Optimization Journey Memory

**Project Rebrand**: TurboQuant → QuantumLeap (March 30, 2026)
- "TurboQuant" remains as the optimization engine name
- Rebranded for better SEO and viral appeal
- "Built on llama.cpp" prominently featured

## 🎯 Final Results (RTX 3050 4GB + i5-11400H + 24GB DDR4)

| Model | Configuration | Speed | Improvement |
|-------|--------------|-------|-------------|
| **MoE 35B-A3B IQ2_XXS** (10GB) | UMA+ngl15+8t+mlock+**no-mmap** | **15.68 tok/s** | **+801% vs baseline** 🏆 |
| Dense 27B Q2_K (9.5GB) | UMA+ngl27+8t+mlock+no-mmap+AVX512 | **4.20 tok/s** | **+141% vs baseline** |
| 4B Q2_K (1.7GB) | Full GPU ngl=99 | **44.80 tok/s** | GPU-bound ceiling |

**Baseline**: Qwen 27B Q4_K_M ngl=4 = 1.74 tok/s

---

## 🔑 Key Discoveries

### 1. **`--no-mmap` Flag = +28% on MoE Models**
**THE BREAKTHROUGH**: Pre-loading MoE models into RAM instead of memory-mapping eliminates page faults during expert routing.

- **Why it works**: MoE models have scattered memory access patterns (different experts activated per token)
- **Memory-mapping**: Causes page faults on each expert switch
- **Pre-loading with --no-mmap**: Eliminates page faults → massive speedup
- **Impact**: 12-13 tok/s → 15-17 tok/s on MoE IQ2_XXS

### 2. **IQ2_XXS Quantization**
- **Size**: 10GB (34% smaller than Q3_K_S 15GB)
- **Quality**: Extreme compression but usable for MoE (3B active params)
- **Benefit**: Smaller size allows more GPU layers (ngl=15 vs ngl=9)
- **IMPORTANT**: IQ quantizations (IQ1_S, IQ2_XXS, IQ2_M, IQ3_XXS, IQ4_NL) **CANNOT be created via requantization**
  - Require importance matrix (imatrix) from original training
  - Must download pre-quantized from HuggingFace (e.g., unsloth, bartowski repos)
  - Safe requant types: Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K only

### 3. **AVX-512 CPU Instructions**
- **Problem**: Original build had AVX/AVX2/AVX-512 ALL disabled
- **Solution**: Rebuilt llama.cpp with full SIMD support
- **Flags**: `GGML_AVX=ON GGML_AVX2=ON GGML_AVX512=ON GGML_AVX512_VBMI=ON GGML_AVX512_VNNI=ON`
- **Impact**: +16% on dense CPU-bound models (3.61 → 4.20 tok/s)
- **Note**: Minimal impact on MoE (memory-bound, not CPU-bound)

### 4. **MoE Architecture Advantage**
- **Qwen3.5-35B-A3B**: 35B total params, only **3B active per token**
- **Result**: 3x faster than dense 27B while having more total intelligence
- **Best for**: 4GB VRAM setups wanting large model quality

---

## 📊 Complete Optimization Progression

```
Baseline (Q4_K_M ngl=4):              1.74 tok/s
↓ Q2_K requantization:                2.82 tok/s (+62%)
↓ +UMA+thread tuning+mlock:           3.61 tok/s (+107%)
↓ +no-mmap+AVX512:                    4.20 tok/s (+141%)
↓ Pivot to MoE (Q3_K_S):             10.73 tok/s (+516%)
↓ MoE IQ2_XXS+no-mmap:               15.68 tok/s (+801%) ✨
```

---

## ⚙️ Auto-Optimization Engine (Integrated in server.py)

All optimizations are **automatically applied** when loading models:

1. **UMA (Unified Memory)** with cliff detection
   - Dense models: cliff at ~42% of total layers
   - MoE models: cliff at ~15% of total layers
   - Pushes ~1.8x more layers to GPU safely

2. **Thread Tuning** (benchmarked on i5-11400H 6C/12T)
   - Dense models: `physical_cores + 2 = 8 threads`
   - MoE models: `physical_cores = 6 threads` (less contention)

3. **mlock** - Always enabled for consistent RAM access

4. **q4_0 KV Cache** - Default compressed attention cache

5. **MoE Detection** - Regex: `\d+[bB][-.]?[aA]\d+[bB]` (e.g., 35B-A3B)

6. **--no-mmap for MoE** - Auto-applied to MoE models (+28% boost)

7. **AVX-512 Support** - Full SIMD instructions enabled in build

---

## 🎚️ UMA NGL Sweet Spots (Benchmarked)

### Dense 27B Q2_K
- ngl=20: 3.18 tok/s
- ngl=22: 3.34 tok/s
- ngl=25: 3.39 tok/s
- **ngl=27: 3.61 tok/s** ← Optimal (without --no-mmap)
- **ngl=27: 4.20 tok/s** ← Optimal (with --no-mmap + AVX512)
- ngl=28: 2.61 tok/s ← **CLIFF**

### MoE 35B-A3B Q3_K_S (15GB)
- ngl=5: 10.02 tok/s
- ngl=8: 10.46 tok/s
- **ngl=9-10: 10.73 tok/s** ← Optimal
- ngl=12: 4.98 tok/s ← **CLIFF**

### MoE 35B-A3B IQ2_XXS (10GB)
- ngl=5: 14.71 tok/s
- ngl=10: 13.81 tok/s
- **ngl=15: 15.68 tok/s** ← Optimal (with --no-mmap)
- ngl=18: 10.98 tok/s ← **CLIFF**
- **Key insight**: Smaller model allows more GPU layers before cliff

---

## 🚫 What Didn't Work

### 1. Speculative Decoding (No improvement)
Tested all variants on 4GB VRAM:
- **Cross-family** (SmolLM draft for Qwen): 2.99 tok/s (no improvement)
- **Same-family** (Qwen3.5-4B for 27B): 2.92 tok/s (VRAM contention hurts)
- **Lookup-based**: 2.91 tok/s (no improvement)
- **MoE + 4B draft**: 8.87 tok/s (worse than MoE alone at 10.73)

**Why it failed**: Draft model VRAM contention on 4GB GPU. The draft model competes for VRAM with the main model, causing slowdowns instead of speedups.

**Conclusion**: Not viable on constrained VRAM. May work on 8GB+ VRAM setups.

### 2. TQ3/TQ4 KV Cache (Not functional)
Tested TurboQuant 3-bit and 4-bit KV cache compression:
- **TQ3 test**: `--cache-type-k tq3 --cache-type-v tq3`
  - Result: **IOT instruction crash** (illegal instruction)
- **TQ4 test**: `--cache-type-k tq4 --cache-type-v tq4`
  - Result: **Fatal error at ggml.c:11796**

**Why it failed**: Types are defined in `ggml.h` (GGML_TYPE_TQ3=500, GGML_TYPE_TQ4=501) but implementation is incomplete or requires special compilation flags.

**Conclusion**: TurboQuant KV cache (the "8x attention speedup" promise) is not functional in current ik_llama.cpp build. May require:
- Different llama.cpp version
- Special CMake flags
- Complete implementation of TurboQuant paper algorithms

### 3. Batch Size Tuning (Minimal impact)
Tested various batch configurations:
- **Batch sizes**: `-b 256/512/1024/2048`
- **Micro-batch sizes**: `-ub 32/64/128/256/512`
- **Result**: Negligible speed difference

**Why it didn't help**: Workload is memory-bound, not batch-bound. DDR4 bandwidth is the bottleneck, not batch processing efficiency.

**Conclusion**: Default batch size is optimal for memory-bound inference.

### 4. Context Size Optimization (Minimal impact)
Tested context sizes: `-c 128/256/512/1024/2048`
- **Result**: Speed dominated by model size, not context length
- **4B model**: ~44-45 tok/s regardless of context size
- **27B model**: ~3.6-4.2 tok/s regardless of context size

**Why it didn't help**: Context affects memory usage but not inference speed on these models. The model weight access dominates computation time.

**Conclusion**: Use the context size needed for your task. No speed penalty for larger contexts (within VRAM limits).

### 5. KV Cache Type Variations (Negligible difference)
Tested on GPU-bound 4B model:
- **f16 KV cache**: ~44-45 tok/s
- **q8_0 KV cache**: ~44-45 tok/s
- **q4_0 KV cache**: ~44-45 tok/s

**Why minimal difference**: GPU-bound models are limited by model weight access, not KV cache access. KV cache is small relative to model size.

**Conclusion**: q4_0 is optimal (good compression, no speed loss, lower memory usage).

### 6. Flash Attention Flag (Already enabled)
- **Status**: Enabled by default in build with `GGML_CUDA_FA_ALL_QUANTS=ON`
- **Test**: Toggling `--flash-attn` flag had no effect
- **Result**: Already at maximum speed

**Conclusion**: Flash Attention is a compile-time optimization, not a runtime flag in this build.

---

## 🧮 Physics Reality Check

### DDR4 Bandwidth Limit (~40 GB/s)
- **Dense 27B Q2_K** (9.5GB model):
  - Theoretical max: ~4.2 tok/s
  - Achieved: 4.20 tok/s = **100% of physics limit** ✅

- **MoE 3B active IQ2_XXS** (0.56GB per token):
  - Theoretical max: ~71 tok/s
  - Achieved: 15.68 tok/s = ~22% (routing overhead + memory access patterns)

### Why 180 tok/s on 27B is Impossible
- 27B Q2_K reads ~9.5GB per token
- At 180 tok/s: 9.5GB × 180 = **1,710 GB/s required**
- DDR4: ~40 GB/s
- **Would need HBM3e** (H100/H200 with ~3,000 GB/s)

---

## 📦 Models on Disk (Final)

### Active Models
- **Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf** (10GB) — Best large model, 15.68 tok/s
- **Qwen3.5-27B.Q2_K.gguf** (9.5GB) — Best dense 27B, 4.20 tok/s
- **Qwen3.5-4B-Q2_K.gguf** (1.7GB) — Fastest single model, 44.80 tok/s

### Deleted (to free space)
- Qwen3.5-35B-A3B-Q3_K_S.gguf (15GB) — replaced by IQ2_XXS
- Qwen3.5-27B.Q4_K_M.gguf (16GB) — baseline, no longer needed
- Qwen3.5-4B-Q4_K_M.gguf (2.6GB) — Q2_K is nearly as fast
- Qwen2.5-0.5B-instruct-q4_k_m.gguf — too small for draft
- SmolLM2-1.7B-Instruct-Q4_K_M.gguf (1GB) — test model

---

## 🔧 Key Files Modified

### `api/server.py` (v0.4.0)
- Added `--no-mmap` flag for MoE models in `_start_llama_server()`
- MoE detection: `is_moe = bool(re.search(r'\d+[bB][-.]?[aA]\d+[bB]', model_path.stem))`
- Thread tuning: `opt_threads = physical_cores if is_moe else min(physical_cores + 2, cpu_count - 1)`
- UMA cliff detection with MoE-specific thresholds

### `README.md`
- Updated benchmark table with IQ2_XXS results
- Added `--no-mmap` to auto-optimization engine list
- Updated hardware compatibility table

### `engine/llama.cpp`
- Rebuilt with AVX-512 support (396 files compiled)
- CMake flags: `GGML_AVX=ON GGML_AVX2=ON GGML_AVX512=ON GGML_AVX512_VBMI=ON GGML_AVX512_VNNI=ON`

### `web/static/app.js` & `web/static/style.css`
- MoE badge display on model cards
- Active params display (e.g., "35B (3B active)")

---

## 📥 How to Download IQ2 Models

IQ quantizations **CANNOT be created via requantization** — they require importance matrix from training.

### Download Sources
1. **unsloth repos**: `https://huggingface.co/unsloth/MODEL-NAME-GGUF`
2. **bartowski repos**: `https://huggingface.co/bartowski/MODEL-NAME-GGUF`

### Example: Qwen3.5-35B-A3B IQ2_XXS
```bash
curl -L -o models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
```

### Available IQ Types
- **IQ1_S, IQ1_M** — 1-bit extreme (quality loss)
- **IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M** — 2-bit extreme
- **IQ3_XXS, IQ3_XS, IQ3_S** — 3-bit
- **IQ4_NL, IQ4_XS** — 4-bit

**Recommendation**: For MoE models, IQ2_XXS or IQ2_M work well due to only 3B active params

---

## 🚀 Next Steps for More Power

If moving to a more powerful PC:

### With 8GB VRAM
- Dense 27B: Full GPU possible with Q4_K_M (~8-10 tok/s)
- MoE 35B-A3B: More GPU layers → 20-25 tok/s possible

### With 12GB+ VRAM
- Dense 70B models with Q2_K/Q3_K
- MoE models fully on GPU → 30-40 tok/s

### With Better CPU (e.g., Ryzen 9 / i9)
- More cores → better thread scaling
- Higher DDR5 bandwidth → 5-6 tok/s on dense 27B

### With DDR5 RAM (~80 GB/s)
- Dense 27B: ~8 tok/s theoretical max
- MoE models: 25-30 tok/s possible

---

## 📝 Lessons Learned

1. **MoE > Dense** for constrained hardware (3x faster for same intelligence)
2. **--no-mmap** is critical for MoE performance (+28%)
3. **IQ2 quantizations** are viable for MoE (3B active params tolerate extreme quant)
4. **Memory bandwidth** is the ultimate bottleneck, not CPU or GPU
5. **UMA cliff detection** is essential (wrong ngl = 50% speed loss)
6. **AVX-512** matters for CPU-bound workloads (+16%)
7. **Speculative decoding** doesn't work on 4GB VRAM (VRAM contention)

---

**Last Updated**: March 30, 2026
**TurboQuant Version**: 0.4.0
**Hardware**: RTX 3050 4GB + i5-11400H + 24GB DDR4
