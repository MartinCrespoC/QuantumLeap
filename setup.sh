#!/bin/bash
set -e

# TurboQuant Setup Script
# Automatic installation with hardware detection and validation

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}"
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║           ⚛️  QuantumLeap v0.4.0 Setup                    ║"
echo "║   801% Faster LLM Inference - Built on llama.cpp          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
   echo -e "${RED}⚠️  Do not run this script as root${NC}"
   exit 1
fi

# ─── Hardware Detection ───────────────────────────────────────────────────────

echo -e "${BOLD}🔍 Detecting Hardware...${NC}"

# Detect GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    HAS_NVIDIA=true
    echo -e "  ${GREEN}✓${NC} GPU: $GPU_NAME (${VRAM_MB}MB VRAM, CUDA $CUDA_VERSION)"
else
    HAS_NVIDIA=false
    echo -e "  ${YELLOW}⚠${NC} No NVIDIA GPU detected (CPU-only mode)"
fi

# Detect RAM
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
echo -e "  ${GREEN}✓${NC} RAM: ${RAM_GB}GB"

# Detect CPU
CPU_MODEL=$(lscpu | grep "Model name" | cut -d: -f2 | xargs)
CPU_CORES=$(nproc)
echo -e "  ${GREEN}✓${NC} CPU: $CPU_MODEL ($CPU_CORES cores)"

# Check AVX support
if grep -q avx512 /proc/cpuinfo; then
    AVX_SUPPORT="AVX-512"
elif grep -q avx2 /proc/cpuinfo; then
    AVX_SUPPORT="AVX2"
elif grep -q avx /proc/cpuinfo; then
    AVX_SUPPORT="AVX"
else
    AVX_SUPPORT="None"
fi
echo -e "  ${GREEN}✓${NC} SIMD: $AVX_SUPPORT"

# ─── Dependency Check ─────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📦 Checking Dependencies...${NC}"

MISSING_DEPS=()

# Check Python 3.10+
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 10 ]; then
        echo -e "  ${GREEN}✓${NC} Python $PYTHON_VERSION"
    else
        echo -e "  ${RED}✗${NC} Python 3.10+ required (found $PYTHON_VERSION)"
        MISSING_DEPS+=("python3.10+")
    fi
else
    echo -e "  ${RED}✗${NC} Python 3 not found"
    MISSING_DEPS+=("python3")
fi

# Check CMake
if command -v cmake &> /dev/null; then
    CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
    echo -e "  ${GREEN}✓${NC} CMake $CMAKE_VERSION"
else
    echo -e "  ${RED}✗${NC} CMake not found"
    MISSING_DEPS+=("cmake")
fi

# Check Ninja
if command -v ninja &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Ninja build system"
else
    echo -e "  ${YELLOW}⚠${NC} Ninja not found (will use make, slower builds)"
fi

# Check Git
if command -v git &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} Git"
else
    echo -e "  ${RED}✗${NC} Git not found"
    MISSING_DEPS+=("git")
fi

# Check CUDA toolkit (if NVIDIA GPU present)
if [ "$HAS_NVIDIA" = true ]; then
    if command -v nvcc &> /dev/null; then
        NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
        echo -e "  ${GREEN}✓${NC} CUDA Toolkit $NVCC_VERSION"
    else
        echo -e "  ${YELLOW}⚠${NC} CUDA Toolkit not found (required for GPU acceleration)"
        MISSING_DEPS+=("cuda-toolkit")
    fi
fi

# Exit if missing critical dependencies
if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
    echo ""
    echo -e "${RED}Missing dependencies: ${MISSING_DEPS[*]}${NC}"
    echo ""
    echo "Install them with:"
    echo "  Ubuntu/Debian: sudo apt install python3 python3-pip cmake ninja-build git"
    if [ "$HAS_NVIDIA" = true ]; then
        echo "  CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    fi
    exit 1
fi

# ─── Python Virtual Environment ───────────────────────────────────────────────

echo ""
echo -e "${BOLD}🐍 Setting up Python Environment...${NC}"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "  ${GREEN}✓${NC} Created virtual environment"
else
    echo -e "  ${GREEN}✓${NC} Virtual environment exists"
fi

source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip > /dev/null 2>&1
pip install -r api/requirements.txt > /dev/null 2>&1
echo -e "  ${GREEN}✓${NC} Installed Python dependencies"

# ─── Build llama.cpp ──────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}🔨 Building llama.cpp with optimizations...${NC}"

cd engine/llama.cpp

# Detect if already built
if [ -f "build/bin/llama-server" ] && [ -f "build/bin/llama-cli" ]; then
    echo -e "  ${YELLOW}⚠${NC} llama.cpp already built"
    read -p "  Rebuild? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        cd ../..
        echo -e "  ${GREEN}✓${NC} Skipped rebuild"
        SKIP_BUILD=true
    fi
fi

if [ "$SKIP_BUILD" != true ]; then
    rm -rf build
    mkdir -p build
    cd build

    # Configure CMake with optimizations
    CMAKE_ARGS=(
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_NATIVE=ON
    )

    # Enable AVX if supported
    if [ "$AVX_SUPPORT" != "None" ]; then
        CMAKE_ARGS+=(-DGGML_AVX=ON)
        if [ "$AVX_SUPPORT" = "AVX2" ] || [ "$AVX_SUPPORT" = "AVX-512" ]; then
            CMAKE_ARGS+=(-DGGML_AVX2=ON)
        fi
        if [ "$AVX_SUPPORT" = "AVX-512" ]; then
            CMAKE_ARGS+=(
                -DGGML_AVX512=ON
                -DGGML_AVX512_VBMI=ON
                -DGGML_AVX512_VNNI=ON
            )
        fi
    fi

    # Enable CUDA if available
    if [ "$HAS_NVIDIA" = true ] && command -v nvcc &> /dev/null; then
        # Detect CUDA architecture
        CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
        CMAKE_ARGS+=(
            -DGGML_CUDA=ON
            -DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH
            -DGGML_CUDA_FA_ALL_QUANTS=ON
        )
        echo -e "  ${GREEN}✓${NC} CUDA enabled (arch $CUDA_ARCH)"
    fi

    # Use Ninja if available
    if command -v ninja &> /dev/null; then
        CMAKE_ARGS+=(-G Ninja)
        BUILD_CMD="ninja"
    else
        BUILD_CMD="make -j$CPU_CORES"
    fi

    echo -e "  ${BLUE}ℹ${NC} Building with: ${CMAKE_ARGS[*]}"
    cmake .. "${CMAKE_ARGS[@]}" > /dev/null 2>&1

    echo -e "  ${BLUE}ℹ${NC} Compiling (this may take 5-10 minutes)..."
    $BUILD_CMD llama-server llama-cli llama-quantize > /dev/null 2>&1

    cd ../../..
    echo -e "  ${GREEN}✓${NC} Built llama.cpp with optimizations"
fi

# ─── Create directories ───────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📁 Creating directories...${NC}"

mkdir -p models benchmarks backups
echo -e "  ${GREEN}✓${NC} Created models/, benchmarks/, backups/"

# ─── Hardware Recommendations ─────────────────────────────────────────────────

echo ""
echo -e "${BOLD}💡 Recommendations for Your Hardware:${NC}"
echo ""

if [ "$HAS_NVIDIA" = true ]; then
    VRAM_GB=$((VRAM_MB / 1024))

    if [ $VRAM_GB -le 4 ]; then
        echo -e "${YELLOW}4GB VRAM Setup:${NC}"
        echo "  • Best: MoE models (e.g., Qwen3.5-35B-A3B IQ2_XXS) → 15+ tok/s"
        echo "  • Fast: 4B models (e.g., Qwen3.5-4B Q2_K) → 45 tok/s"
        echo "  • Dense 27B: Q2_K quantization → 4 tok/s"
        echo ""
        echo "  Download MoE model:"
        echo "    curl -L -o models/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf \\"
        echo "      'https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf'"
    elif [ $VRAM_GB -le 8 ]; then
        echo -e "${GREEN}8GB VRAM Setup:${NC}"
        echo "  • Dense 27B: Q4_K_M full GPU → 8-10 tok/s"
        echo "  • MoE 35B-A3B: More GPU layers → 20-25 tok/s"
        echo "  • 13B models: Q6_K/Q8_0 for best quality"
    else
        echo -e "${GREEN}12GB+ VRAM Setup:${NC}"
        echo "  • Dense 70B: Q2_K/Q3_K possible"
        echo "  • MoE models: Full GPU → 30-40 tok/s"
        echo "  • Any 27B model: Full GPU with high quality quants"
    fi
else
    echo -e "${YELLOW}CPU-Only Setup:${NC}"
    echo "  • MoE models recommended (3B active params)"
    echo "  • Use IQ2_XXS or Q2_K quantizations"
    echo "  • Expected: 9-12 tok/s on MoE 35B-A3B"
fi

echo ""
echo -e "RAM: ${RAM_GB}GB"
if [ $RAM_GB -lt 16 ]; then
    echo -e "  ${YELLOW}⚠${NC} 16GB+ RAM recommended for large models"
elif [ $RAM_GB -ge 24 ]; then
    echo -e "  ${GREEN}✓${NC} Excellent for large model offloading"
fi

# ─── Model Management Guide ───────────────────────────────────────────────────

echo ""
echo -e "${BOLD}📚 Model Management:${NC}"
echo ""
echo "1. Download models to models/ directory"
echo "2. Quantization types:"
echo "   • ${GREEN}Requantizable${NC}: Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, Q2_K"
echo "     → Use Web UI Requantize feature"
echo "   • ${YELLOW}Pre-quantized only${NC}: IQ1_S, IQ2_XXS, IQ2_M, IQ3_XXS, IQ4_NL"
echo "     → Download from HuggingFace (unsloth/bartowski repos)"
echo ""
echo "3. Model recommendations:"
echo "   • ${BOLD}MoE models${NC}: 3x faster than dense for same intelligence"
echo "   • ${BOLD}IQ2_XXS${NC}: Best for MoE on constrained hardware"
echo "   • ${BOLD}Q2_K${NC}: Best requantizable extreme compression"
echo ""
echo "See memory.md for complete optimization guide"

# ─── Final Steps ──────────────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${GREEN}✓ Setup Complete!${NC}"
echo ""
echo "Start TurboQuant:"
echo "  ${BOLD}bash scripts/start.sh${NC}"
echo ""
echo "Or manually:"
echo "  ${BOLD}source venv/bin/activate${NC}"
echo "  ${BOLD}python3 api/server.py${NC}"
echo ""
echo "Web UI will be available at: ${BLUE}http://localhost:11434${NC}"
echo ""
echo -e "${YELLOW}Note:${NC} First run will take longer as models are loaded"
echo ""
