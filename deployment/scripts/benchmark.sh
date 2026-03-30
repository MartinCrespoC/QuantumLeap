#!/bin/bash
# TurboQuant Benchmark Suite
set -euo pipefail

echo "=== TurboQuant Benchmark Suite ==="
echo "Date: $(date)"
echo "Kernel: $(uname -r)"
echo ""

# Hardware info
echo "--- Hardware ---"
echo "CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)"
echo "RAM: $(free -h | awk '/Mem:/ {print $2}')"
if command -v nvidia-smi &>/dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader)"
fi
echo "AVX-512: $(grep -q avx512f /proc/cpuinfo && echo 'YES' || echo 'NO')"
echo ""

# Run C++ benchmarks if built
CORE_BIN="../core/build"
if [ -f "$CORE_BIN/benchmark" ]; then
    echo "--- C++ Benchmarks ---"
    "$CORE_BIN/benchmark"
    echo ""
fi

# Run accuracy tests
if [ -f "$CORE_BIN/test_all" ]; then
    echo "--- Accuracy Tests ---"
    "$CORE_BIN/test_all"
    echo ""
fi

# Python benchmarks
if [ -d "../python/.venv" ]; then
    echo "--- Python Benchmarks ---"
    source ../python/.venv/bin/activate
    python ../python/benchmarks/speed_test.py 2>/dev/null || echo "Python benchmarks not ready"
    echo ""
fi

# Ollama benchmarks (if installed)
if command -v ollama &>/dev/null; then
    echo "--- Ollama Model Benchmarks ---"
    for model in llama3.2:3b mistral:7b; do
        if ollama list 2>/dev/null | grep -q "$model"; then
            echo "Model: $model"
            echo "prompt eval test" | timeout 60 ollama run "$model" --verbose 2>&1 | grep -E "eval|token" || true
            echo ""
        fi
    done
fi

echo "=== Benchmark Complete ==="
