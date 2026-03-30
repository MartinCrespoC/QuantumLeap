#!/bin/bash
# QuantumLeap Quick Start — macOS (Apple Silicon + Intel)
# 801% faster LLM inference built on llama.cpp
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo ""
echo "  ⚛️  QuantumLeap v0.4.0 — 801% Faster LLM Inference"
echo "  ══════════════════════════════════════════════════"
echo "  Built on llama.cpp | Powered by TurboQuant Engine"
echo ""

# Check engine
if [ ! -f "engine/llama.cpp/build/bin/llama-server" ]; then
  echo "  ❌ Engine not built. Run: bash scripts/setup.sh"
  exit 1
fi

# Detect hardware (macOS)
if sysctl -n machdep.cpu.brand_string &>/dev/null; then
  CPU=$(sysctl -n machdep.cpu.brand_string 2>/dev/null)
  echo "  CPU: ${CPU}"
fi
RAM_GB=$(( $(sysctl -n hw.memsize 2>/dev/null || echo 0) / 1073741824 ))
echo "  RAM: ${RAM_GB} GB"
# Check for Metal (Apple Silicon)
if system_profiler SPDisplaysDataType 2>/dev/null | grep -q "Metal"; then
  echo "  GPU: Metal supported (Apple Silicon / AMD)"
fi
echo ""

# Python venv
if [ ! -d ".venv" ] && [ ! -d "venv" ]; then
  echo "  📦 Creating Python virtual environment..."
  python3 -m venv .venv
fi

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
fi

if ! python3 -c "import fastapi" 2>/dev/null; then
  echo "  📦 Installing Python dependencies..."
  pip install --quiet -r api/requirements.txt
fi

MODEL_COUNT=$(find models/ -name "*.gguf" 2>/dev/null | wc -l | tr -d ' ')
if [ "$MODEL_COUNT" -eq 0 ]; then
  echo "  ⚠️  No models found in models/"
  echo "  💡 Use the Web UI to search and download models"
  echo ""
else
  echo "  📂 Models: ${MODEL_COUNT} GGUF files found"
fi

echo ""
echo "  ✅ Starting TurboQuant Server..."
echo ""
echo "  🌐 Web UI:     http://localhost:${API_PORT:-11434}"
echo "  🔌 Ollama API: http://localhost:${API_PORT:-11434}/api/"
echo "  🤖 OpenAI API: http://localhost:${API_PORT:-11434}/v1/"
echo "  📊 Features:   Auto-offloading, Requantization, Smart Search"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

exec python3 api/server.py "$@"
