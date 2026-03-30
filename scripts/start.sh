#!/bin/bash
# QuantumLeap Quick Start — 801% faster LLM inference built on llama.cpp
# Works on Linux. See start.bat for Windows, start_mac.sh for macOS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

export PATH="/opt/cuda/bin:/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/opt/cuda/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

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

# Detect hardware
if command -v nvidia-smi &>/dev/null; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  echo "  GPU: ${GPU_NAME} (${VRAM} MB VRAM)"
else
  echo "  GPU: Not detected (CPU-only mode)"
fi
RAM_MB=$(free -m 2>/dev/null | awk '/^Mem:/{print $2}' || echo "unknown")
echo "  RAM: ${RAM_MB} MB"
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

# Check models
MODEL_COUNT=$(find models/ -name "*.gguf" 2>/dev/null | wc -l)
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
echo "  📊 Features:   Auto-offloading, UMA, Requantization, Smart Search"
echo ""
echo "  Press Ctrl+C to stop"
echo ""

exec python3 api/server.py "$@"
