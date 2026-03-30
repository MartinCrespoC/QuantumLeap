# Contributing to QuantumLeap

Thank you for your interest in contributing to QuantumLeap! This document provides guidelines and information for contributors.

**QuantumLeap** is built on [llama.cpp](https://github.com/ggerganov/llama.cpp) with the TurboQuant optimization engine.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/quantumleap.git`
3. Run setup: `bash setup.sh`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Format with `black` (100 char line length)
- Docstrings for all public functions

### C++/CUDA
- C++20 standard
- Google Style Guide
- clang-format enforced
- Intel syntax for assembly (`.intel_syntax noprefix`)
- Comment every assembly instruction

### Commit Messages
Use conventional commits format:
- `feat:` New features
- `fix:` Bug fixes
- `perf:` Performance improvements
- `docs:` Documentation changes
- `test:` Test additions/changes
- `refactor:` Code refactoring

Example: `feat: add --no-mmap optimization for MoE models`

## Performance Requirements

All performance-critical changes must include:
- Benchmark results before/after
- Profiling data (`perf` or `nvprof`)
- Minimum 5% improvement for optimization PRs

### Benchmarking
```bash
# Use llama-cli for consistent benchmarks
engine/llama.cpp/build/bin/llama-cli \
  -m models/YOUR_MODEL.gguf \
  -ngl 99 -c 512 -n 128 \
  -p "Test prompt" \
  --no-display-prompt
```

## Testing

### Unit Tests
- All quantization functions must have tests
- Accuracy < 1% degradation vs FP16
- Memory leak checks with Valgrind

### Integration Tests
- Test with llama.cpp GGML backend
- Verify API endpoints work
- Check Web UI functionality

## Pull Request Process

1. Update documentation (README.md, memory.md if applicable)
2. Add tests for new features
3. Ensure all tests pass
4. Include benchmark results for performance changes
5. Update CHANGELOG.md
6. Request review from maintainers

## Areas for Contribution

### High Priority
- Additional quantization methods (GPTQ, AWQ support)
- More MoE model optimizations
- Apple Silicon Metal backend improvements
- Windows CUDA optimization

### Medium Priority
- Additional model architectures (Mixtral, DeepSeek-V2)
- Batch processing improvements
- API rate limiting and authentication
- Docker containerization

### Documentation
- Translation to other languages
- Video tutorials
- Model compatibility database
- Hardware compatibility matrix expansion

## Hardware for Testing

If contributing performance optimizations, please test on:
- **Minimum**: 4GB VRAM GPU (RTX 3050 class)
- **Recommended**: Multiple GPU tiers (4GB, 8GB, 12GB+)
- **CPU**: Both Intel (AVX-512) and AMD (AVX2)

## Questions?

- Open an issue for bugs or feature requests
- Join discussions for questions
- Check `memory.md` for optimization details

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
