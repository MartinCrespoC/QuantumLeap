#!/bin/bash
# Download GGUF models for TurboQuant
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$(dirname "$SCRIPT_DIR")/models"
mkdir -p "$MODELS_DIR"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

download() {
  local name="$1" url="$2" file="$MODELS_DIR/$3"
  if [ -f "$file" ]; then
    echo -e "${GREEN}[✓]${NC} $name already downloaded"
    return
  fi
  echo -e "${CYAN}[↓]${NC} Downloading $name..."
  if command -v wget &>/dev/null; then
    wget -q --show-progress -O "$file" "$url"
  else
    curl -L --progress-bar -o "$file" "$url"
  fi
  echo -e "${GREEN}[✓]${NC} $name ($(du -h "$file" | cut -f1))"
}

echo ""
echo "  ⚡ TurboQuant Model Downloader"
echo ""

case "${1:-smollm}" in
  smollm|1b|small|test)
    download "SmolLM2 1.7B Instruct Q4_K_M" \
      "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf" \
      "SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
    ;;
  qwen|3b)
    download "Qwen2.5 3B Instruct Q4_K_M" \
      "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf" \
      "qwen2.5-3b-instruct-q4_k_m.gguf"
    ;;
  llama|8b)
    download "Llama 3.2 3B Instruct Q4_K_M" \
      "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
      "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    ;;
  all)
    $0 smollm
    $0 qwen
    $0 llama
    ;;
  *)
    echo "Usage: $0 [smollm|qwen|llama|all]"
    echo ""
    echo "  smollm  SmolLM2 1.7B (Q4_K_M, ~1GB)   — Fast testing"
    echo "  qwen    Qwen2.5 3B  (Q4_K_M, ~2GB)     — Good quality"
    echo "  llama   Llama3.2 3B (Q4_K_M, ~2GB)      — Best balance"
    echo "  all     Download all"
    ;;
esac

echo ""
echo "Models directory: $MODELS_DIR"
ls -lh "$MODELS_DIR"/*.gguf 2>/dev/null || echo "No models found"
