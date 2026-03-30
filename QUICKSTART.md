# QuantumLeap Quick Start Guide

Get QuantumLeap running in 5 minutes with automatic hardware detection and optimization.

**Built on llama.cpp** with TurboQuant optimization engine for 801% faster inference.

## Prerequisites

- **Linux/macOS/Windows** with Python 3.10+
- **4GB+ RAM** (16GB+ recommended for large models)
- **Optional**: NVIDIA GPU with 4GB+ VRAM (CPU-only works too)

## Installation

### 1. Clone and Setup

```bash
git clone https://github.com/YOUR_USERNAME/quantumleap.git
cd quantumleap
bash setup.sh
```

The setup script automatically:
- ✅ Detects your GPU, RAM, CPU, and SIMD support
- ✅ Installs Python dependencies in virtual environment
- ✅ Builds llama.cpp with AVX-512/CUDA optimizations
- ✅ Creates necessary directories
- ✅ Provides hardware-specific recommendations

**Setup takes 5-10 minutes** (mostly compiling llama.cpp)

### 2. Download a Model

**For 4GB VRAM (RTX 3050 / GTX 1650):**
```bash
# Best: MoE model - 15+ tok/s, 35B intelligence, 3B active
curl -L -o models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \
  "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf"
```

**For 8GB+ VRAM:**
```bash
# Dense model with better quality
curl -L -o models/Qwen3.5-27B-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen3.5-27B-GGUF/resolve/main/qwen3.5-27b-q4_k_m.gguf"
```

**For CPU-Only:**
```bash
# Small fast model
curl -L -o models/Qwen3.5-4B-Q2_K.gguf \
  "https://huggingface.co/Qwen/Qwen3.5-4B-GGUF/resolve/main/qwen3.5-4b-q2_k.gguf"
```

### 3. Start the Server

```bash
bash scripts/start.sh
```

**That's it!** Open http://localhost:11434 in your browser.

## First Steps

### Web UI (Recommended)

1. Open http://localhost:11434
2. Go to **Models** tab
3. Your downloaded model should appear
4. Click **Load** to start using it
5. Go to **Chat** tab to start chatting

### API Usage

**Ollama-compatible:**
```bash
curl http://localhost:11434/api/chat -d '{
  "model": "Qwen3.5-35B-A3B-UD-IQ2_XXS",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

**OpenAI-compatible:**
```bash
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.5-35B-A3B-UD-IQ2_XXS",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Connect from IDE

**Windsurf / VSCode / Cursor:**
- Set Ollama endpoint to: `http://localhost:11434`
- Model will auto-load on first request

## Troubleshooting

### "Model not found"
- Check `models/` directory has `.gguf` files
- Restart server: `bash scripts/start.sh`

### "Out of memory" / Crash
- Model too large for your hardware
- Try smaller quantization (Q2_K instead of Q4_K_M)
- Or use MoE model (3B active params vs 27B dense)

### Slow performance
- Check `memory.md` for optimization tips
- Ensure AVX-512 is enabled (rebuild with `bash setup.sh`)
- Try `--no-mmap` flag (auto-enabled for MoE)

### CUDA not detected
- Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads
- Rebuild: `cd engine/llama.cpp/build && rm -rf * && cd ../../.. && bash setup.sh`

### Python version issues
```bash
# Use Python 3.10+ explicitly
python3.10 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt
```

## Next Steps

- **Model Management**: See [README.md#model-management-guide](README.md#model-management-guide)
- **Optimization Details**: Read `memory.md` for all benchmarks and findings
- **Requantization**: Use Web UI to convert models to smaller sizes
- **Smart Search**: Find HuggingFace models optimized for your hardware

## Performance Expectations (RTX 3050 4GB + i5-11400H)

### 4GB VRAM (RTX 3050)
- MoE 35B-A3B IQ2_XXS: **15+ tok/s**
- Dense 27B Q2_K: **4 tok/s**
- 4B Q2_K: **45 tok/s**

### 8GB VRAM (RTX 3060)
- Dense 27B Q4_K_M: **8-10 tok/s**
- MoE 35B-A3B Q3_K_S: **20-25 tok/s**

### CPU-Only
- MoE 35B-A3B IQ2_XXS: **9-12 tok/s**
- 4B Q2_K: **15-20 tok/s**

## Getting Help

- **Issues**: Open a GitHub issue
- **Discussions**: Use GitHub Discussions
- **Documentation**: Check `memory.md` and `CONTRIBUTING.md`

---

**Ready to optimize further?** Check out the [Model Management Guide](README.md#model-management-guide) for advanced tips!
