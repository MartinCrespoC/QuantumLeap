#!/bin/bash
# TurboQuant Model Launcher
# Intelligent launcher that auto-configures GPU/CPU offloading
# based on model size and available VRAM

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
MODELS_DIR="${HOME}/.cache/turboquant/models"
DEFAULT_PORT=8000

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
NC='\033[0m'

usage() {
    echo -e "${CYAN}TurboQuant Model Launcher${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS] [MODEL]"
    echo ""
    echo "Options:"
    echo "  --api          Start as OpenAI-compatible API server"
    echo "  --webui        Start Open WebUI"
    echo "  --cli          Interactive CLI mode (default)"
    echo "  --model NAME   Model to load (e.g., llama-3.2-3b, mistral-7b)"
    echo "  --bits N       Quantization bits: 2 or 4 (default: auto)"
    echo "  --port PORT    API server port (default: 8000)"
    echo "  --gpu-layers N Override GPU layer count"
    echo "  --benchmark    Run benchmark instead of inference"
    echo "  --help         Show this help"
    echo ""
    echo "Models:"
    echo "  llama-3.2-3b   Llama 3.2 3B (fits fully in 4GB VRAM)"
    echo "  mistral-7b     Mistral 7B (fits in VRAM with TQ2)"
    echo "  llama-3.1-13b  Llama 3.1 13B (partial GPU offload)"
    echo "  qwen-72b       Qwen 72B (mostly CPU, partial GPU)"
}

# Detect available VRAM
detect_vram() {
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
    else
        echo "0"
    fi
}

# Auto-calculate GPU layers based on model and VRAM
auto_gpu_layers() {
    local model="$1"
    local vram_mb="$2"
    local bits="${3:-2}"

    # Reserve 500MB for overhead
    local usable=$((vram_mb - 500))
    [ $usable -lt 0 ] && usable=0

    case "$model" in
        llama-3.2-3b|llama3.2-3b)
            if [ "$bits" -eq 2 ]; then
                echo "99"  # Full GPU with TQ2 (~800MB)
            else
                echo "99"  # Full GPU with TQ4 (~1.5GB)
            fi
            ;;
        mistral-7b)
            if [ "$bits" -eq 2 ]; then
                echo "99"  # Full GPU with TQ2 (~1.9GB)
            else
                echo "$((usable / 120))"  # ~120MB per layer with TQ4
            fi
            ;;
        llama-3.1-13b|llama3.1-13b)
            echo "$((usable / 100))"  # ~100MB per layer with TQ2
            ;;
        qwen-72b)
            echo "$((usable / 180))"  # ~180MB per layer with TQ2
            ;;
        *)
            echo "$((usable / 120))"  # Conservative default
            ;;
    esac
}

# Main
MODE="cli"
MODEL=""
BITS="auto"
PORT=$DEFAULT_PORT
GPU_LAYERS=""
BENCHMARK=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --api) MODE="api"; shift ;;
        --webui) MODE="webui"; shift ;;
        --cli) MODE="cli"; shift ;;
        --model) MODEL="$2"; shift 2 ;;
        --bits) BITS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --gpu-layers) GPU_LAYERS="$2"; shift 2 ;;
        --benchmark) BENCHMARK=true; shift ;;
        --help) usage; exit 0 ;;
        *) MODEL="$1"; shift ;;
    esac
done

# Defaults
[ -z "$MODEL" ] && MODEL="llama-3.2-3b"

# Detect hardware
VRAM_MB=$(detect_vram)
echo -e "${CYAN}=== TurboQuant Launcher ===${NC}"
echo -e "Model:     ${GREEN}$MODEL${NC}"
echo -e "VRAM free: ${GREEN}${VRAM_MB}MB${NC}"

# Auto-select bits
if [ "$BITS" = "auto" ]; then
    if [ "$VRAM_MB" -lt 2000 ]; then
        BITS=2
    else
        BITS=4
    fi
fi
echo -e "Quant:     ${GREEN}INT${BITS}${NC}"

# Auto-calculate GPU layers
if [ -z "$GPU_LAYERS" ]; then
    GPU_LAYERS=$(auto_gpu_layers "$MODEL" "$VRAM_MB" "$BITS")
fi
echo -e "GPU layers: ${GREEN}${GPU_LAYERS}${NC}"

# Check if ollama is available
if ! command -v ollama &>/dev/null; then
    echo -e "${YELLOW}Ollama not installed. Install with:${NC}"
    echo "  curl -fsSL https://ollama.ai/install.sh | sh"
    exit 1
fi

case "$MODE" in
    cli)
        echo -e "\n${GREEN}Starting interactive CLI...${NC}"
        OLLAMA_NUM_GPU=$GPU_LAYERS ollama run "$MODEL"
        ;;
    api)
        echo -e "\n${GREEN}Starting API server on port $PORT...${NC}"
        OLLAMA_NUM_GPU=$GPU_LAYERS OLLAMA_HOST="0.0.0.0:$PORT" ollama serve &
        sleep 2
        echo -e "${GREEN}API ready at http://localhost:$PORT${NC}"
        echo -e "Test: curl http://localhost:$PORT/api/generate -d '{\"model\":\"$MODEL\",\"prompt\":\"Hello\"}'"
        wait
        ;;
    webui)
        echo -e "\n${GREEN}Starting Open WebUI...${NC}"
        if command -v open-webui &>/dev/null; then
            OLLAMA_NUM_GPU=$GPU_LAYERS ollama serve &
            sleep 2
            open-webui serve --port 3000
        else
            echo -e "${YELLOW}Open WebUI not installed. Install with:${NC}"
            echo "  pip install open-webui"
        fi
        ;;
esac
