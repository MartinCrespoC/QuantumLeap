---
description: How to integrate TurboQuant with an LLM model for inference
---

# Skill: Integrate TurboQuant with Model

## Steps
1. Export model to safetensors or GGUF format
2. Run calibration to collect activation statistics:
   ```bash
   python -m turboquant_py.training.calibration --model <path>
   ```
3. Apply TurboQuant quantization:
   ```python
   from turboquant_py import TurboQuantizer, QuantConfig, QuantBits
   config = QuantConfig(bits=QuantBits.INT2, group_size=128)
   q = TurboQuantizer(config)
   result = q.quantize_tensor(weight_tensor)
   ```
4. Verify accuracy: compare perplexity vs FP16 baseline
5. Benchmark inference speed with `./benchmark`

## Expected results
- Size: 2-4x smaller than Q4_K_M
- Speed: 1.4-1.8x faster than Q4_K_M
- Accuracy: >98.5% vs FP16

## Model size guide (TurboQuant INT2)
| Model | FP16 | Q4_K_M | TQ2 | Fits 4GB? |
|-------|------|--------|-----|-----------|
| 3B | 6 GB | 1.8 GB | 0.8 GB | ✅ |
| 7B | 14 GB | 4.2 GB | 1.9 GB | ✅ |
| 13B | 26 GB | 7.4 GB | 3.5 GB | ✅ |
| 70B | 140 GB | 40 GB | 18.6 GB | ❌ (offload) |
