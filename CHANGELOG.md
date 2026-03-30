# Changelog

All notable changes to QuantumLeap will be documented in this file.

**Project Rebrand**: TurboQuant → QuantumLeap (v0.4.0) - "TurboQuant" remains as the optimization engine name.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-03-30

### Added
- **Project rebrand**: TurboQuant → QuantumLeap (TurboQuant remains as engine name)
- **--no-mmap optimization** for MoE models (+28% speed boost)
- **"Optimization Journey" section** in README documenting all tested optimizations
- **"Best Practices" section** with backup recommendations and decision trees
- Automatic MoE detection and optimization in `_start_llama_server()`
- MoE badge display in Web UI model cards
- Active parameters display for MoE models (e.g., "35B (3B active)")
- Comprehensive Model Management Guide in README
- Hardware-specific model recommendations in setup script
- `memory.md` with complete optimization journey documentation
- `setup.sh` automated setup with hardware detection
- `.gitignore`, `LICENSE`, `CONTRIBUTING.md` for GitHub
- SEO optimization: "Built on llama.cpp" prominently featured

### Changed
- Rebuilt llama.cpp with full AVX-512 support (+16% on dense models)
- Updated benchmark results with IQ2_XXS MoE performance
- Thread tuning: 8 threads for dense, 6 for MoE models
- UMA cliff detection: 42% for dense, 15% for MoE models
- README restructured with Quick Start and Model Management sections

### Performance
- **MoE 35B-A3B IQ2_XXS**: 15.68 tok/s (+801% vs baseline)
- **Dense 27B Q2_K**: 4.20 tok/s (+141% vs baseline)
- **4B Q2_K**: 44.80 tok/s (GPU-bound ceiling)

### Fixed
- AVX/AVX2/AVX-512 were disabled in previous build (now enabled)
- MoE models now use optimal thread count (6 vs 8)
- UMA cliff detection prevents performance drops

## [0.3.0] - 2026-03-29

### Added
- Auto-offloading with CUDA Unified Memory (UMA)
- Web UI Compatibility Calculator
- Requantization API with real-time progress tracking
- Smart Search for hardware-optimized HuggingFace models
- Cross-platform support (Linux, Windows, macOS)
- TurboQuant KV cache compression (q4_0)

### Performance
- Q2_K quantization: 2.82 tok/s on 27B (+62% vs baseline)
- UMA + thread tuning: 3.61 tok/s (+107% vs baseline)
- MoE Q3_K_S: 10.73 tok/s (+516% vs baseline)

### Changed
- Backend API restructured with FastAPI
- Web UI redesigned with dark theme
- Model manager with compatibility badges

## [0.2.0] - 2026-03-28

### Added
- Basic auto-offloading for GPU+CPU mixed inference
- Hardware detection via nvidia-smi
- Ollama-compatible API endpoints
- OpenAI-compatible API endpoints

### Performance
- Baseline Q4_K_M: 1.74 tok/s on 27B

## [0.1.0] - 2026-03-27

### Added
- Initial release
- Basic llama.cpp integration
- Simple web interface
- Model loading and inference

---

## Upgrade Guide

### 0.3.0 → 0.4.0

**Breaking Changes**: None

**Recommended Actions**:
1. Rebuild llama.cpp with AVX-512: `bash setup.sh` (or rebuild manually)
2. Download IQ2_XXS MoE models for best performance
3. Update `server.py` if you have local modifications (check `--no-mmap` integration)

**New Features**:
- MoE models automatically use `--no-mmap` for +28% speed
- Setup script now detects hardware and provides recommendations
- Model Management Guide added to README

### 0.2.0 → 0.3.0

**Breaking Changes**: None

**Recommended Actions**:
1. Install new Python dependencies: `pip install -r api/requirements.txt`
2. Clear old model cache if experiencing issues
3. Re-download models if using BitNet (not supported via requantization)

**New Features**:
- Requantization from Web UI
- Smart Search with hardware compatibility
- UMA auto-enabled for mixed offloading
