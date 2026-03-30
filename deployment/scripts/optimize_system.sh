#!/bin/bash
# System optimizations specific to LLM inference workloads
# Run with sudo

set -euo pipefail

echo "=== TurboQuant System Optimizer ==="

# Memory management for large model allocations
echo "[1/6] Configuring memory management..."
sysctl -w vm.max_map_count=262144
sysctl -w vm.overcommit_memory=1
sysctl -w vm.swappiness=5
sysctl -w vm.dirty_ratio=20

# Transparent Huge Pages (better for large contiguous allocations)
echo madvise > /sys/kernel/mm/transparent_hugepage/enabled

# NVIDIA GPU optimization
echo "[2/6] Configuring GPU..."
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi -pm 1                    # Persistence mode
    nvidia-smi -pl 60 2>/dev/null || true  # Power limit (max for RTX 3050 Laptop)
    nvidia-smi --lock-gpu-clocks=1500,1500 2>/dev/null || true  # Lock to max clock
fi

# CUDA environment
echo "[3/6] Setting CUDA environment..."
cat > /etc/profile.d/turboquant-cuda.sh << 'CUDA_EOF'
export CUDA_DEVICE_MAX_CONNECTIONS=32
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export OMP_NUM_THREADS=12
export OMP_PROC_BIND=close
export OMP_PLACES=cores
CUDA_EOF

# CPU performance
echo "[4/6] Setting CPU performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
    echo performance > "$cpu" 2>/dev/null || true
done

# I/O optimization for model loading
echo "[5/6] Optimizing I/O..."
echo 2 > /proc/sys/vm/dirty_expire_centisecs
echo 4096 > /sys/block/nvme0n1/queue/read_ahead_kb 2>/dev/null || true

# Network (for API server)
echo "[6/6] Optimizing network..."
sysctl -w net.core.somaxconn=4096
sysctl -w net.ipv4.tcp_max_syn_backlog=4096

echo ""
echo "=== System optimized for LLM inference ==="
echo "  Memory: max_map_count=262144, overcommit=1, swappiness=5"
echo "  GPU: Persistence mode, max power, locked clocks"
echo "  CPU: Performance EPP, 12 OMP threads"
echo "  I/O: 4MB readahead for NVMe"
