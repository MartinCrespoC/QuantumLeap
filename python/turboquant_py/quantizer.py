"""TurboQuant Quantizer: Python API for extreme model quantization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


class QuantBits(IntEnum):
    INT2 = 2
    INT4 = 4
    INT8 = 8


@dataclass
class QuantConfig:
    """Configuration for TurboQuant quantization."""

    bits: QuantBits = QuantBits.INT2
    group_size: int = 128
    num_residual_iterations: int = 3
    use_polar_transform: bool = True
    calibration_samples: int = 512
    device: str = "cuda"
    vram_budget_mb: int = 3500  # Leave ~500MB for system on 4GB GPU


@dataclass
class QuantResult:
    """Result of quantization with metrics."""

    packed_data: bytes
    scales: np.ndarray
    zero_points: np.ndarray
    original_shape: tuple[int, ...]
    bits: QuantBits
    mse: float = 0.0
    max_error: float = 0.0
    compression_ratio: float = 0.0


class TurboQuantizer:
    """Main quantizer class implementing TurboQuant algorithm."""

    def __init__(self, config: Optional[QuantConfig] = None) -> None:
        self.config = config or QuantConfig()
        self._check_hardware()

    def _check_hardware(self) -> None:
        """Detect and report available hardware."""
        self.has_cuda = torch.cuda.is_available()
        self.has_avx512 = self._detect_avx512()

        if self.has_cuda:
            self.gpu_name = torch.cuda.get_device_name(0)
            self.vram_total = torch.cuda.get_device_properties(0).total_memory
            self.vram_free = torch.cuda.mem_get_info()[0]
        else:
            self.gpu_name = "N/A"
            self.vram_total = 0
            self.vram_free = 0

    def _detect_avx512(self) -> bool:
        """Check if CPU supports AVX-512."""
        try:
            with open("/proc/cpuinfo", "r") as f:
                cpuinfo = f.read()
            return "avx512f" in cpuinfo
        except FileNotFoundError:
            return False

    def print_hardware_info(self) -> None:
        """Print detected hardware capabilities."""
        table = Table(title="TurboQuant Hardware Detection")
        table.add_column("Feature", style="cyan")
        table.add_column("Status", style="green")

        table.add_row("CUDA", "✅ Available" if self.has_cuda else "❌ Not found")
        table.add_row("GPU", self.gpu_name)
        table.add_row(
            "VRAM",
            f"{self.vram_total / (1024**3):.1f} GB"
            + f" ({self.vram_free / (1024**3):.1f} GB free)"
            if self.has_cuda
            else "N/A",
        )
        table.add_row("AVX-512", "✅ Supported" if self.has_avx512 else "❌ Not found")
        table.add_row("Quant Bits", f"INT{self.config.bits}")
        table.add_row("Group Size", str(self.config.group_size))

        console.print(table)

    def quantize_tensor(self, tensor: torch.Tensor) -> QuantResult:
        """Quantize a single tensor using TurboQuant.

        Args:
            tensor: Input FP32/FP16 tensor to quantize.

        Returns:
            QuantResult with packed data and metadata.
        """
        data = tensor.float().cpu().numpy().flatten()
        n = len(data)
        max_quant = (1 << int(self.config.bits)) - 1

        # Group-wise quantization
        group_size = self.config.group_size
        num_groups = (n + group_size - 1) // group_size
        scales = np.zeros(num_groups, dtype=np.float32)
        zero_points = np.zeros(num_groups, dtype=np.float32)
        quantized = np.zeros(n, dtype=np.int32)

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, n)
            group = data[start:end]

            min_val = group.min()
            max_val = group.max()
            range_val = max_val - min_val

            if range_val < 1e-8:
                scales[g] = 1.0
                zero_points[g] = 0.0
                continue

            scale = range_val / max_quant
            scales[g] = scale
            zero_points[g] = min_val

            q = np.round((group - min_val) / scale).astype(np.int32)
            q = np.clip(q, 0, max_quant)
            quantized[start:end] = q

        # Pack bits
        if self.config.bits == QuantBits.INT2:
            packed = self._pack_int2(quantized)
        elif self.config.bits == QuantBits.INT4:
            packed = self._pack_int4(quantized)
        else:
            packed = quantized.astype(np.uint8).tobytes()

        # Compute error
        dequantized = np.zeros(n, dtype=np.float32)
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, n)
            dequantized[start:end] = quantized[start:end] * scales[g] + zero_points[g]

        mse = float(np.mean((data - dequantized) ** 2))
        max_err = float(np.max(np.abs(data - dequantized)))

        return QuantResult(
            packed_data=packed,
            scales=scales,
            zero_points=zero_points,
            original_shape=tuple(tensor.shape),
            bits=self.config.bits,
            mse=mse,
            max_error=max_err,
            compression_ratio=n * 4 / len(packed),
        )

    def _pack_int2(self, values: np.ndarray) -> bytes:
        """Pack INT2 values: 4 values per byte."""
        n = len(values)
        packed = bytearray((n + 3) // 4)
        for i in range(n):
            packed[i // 4] |= (int(values[i]) & 0x3) << ((i % 4) * 2)
        return bytes(packed)

    def _pack_int4(self, values: np.ndarray) -> bytes:
        """Pack INT4 values: 2 values per byte."""
        n = len(values)
        packed = bytearray((n + 1) // 2)
        for i in range(0, n, 2):
            lo = int(values[i]) & 0xF
            hi = (int(values[i + 1]) & 0xF) if (i + 1 < n) else 0
            packed[i // 2] = lo | (hi << 4)
        return bytes(packed)

    def estimate_model_size(self, param_count: int) -> None:
        """Print estimated model sizes for different quantization methods."""
        table = Table(title=f"Model Size Estimate ({param_count / 1e9:.1f}B params)")
        table.add_column("Method", style="cyan")
        table.add_column("Size", style="green")
        table.add_column("Fits in 4GB VRAM?", style="yellow")

        methods = [
            ("FP16", param_count * 2),
            ("Q8_0", param_count * 1),
            ("Q4_K_M", int(param_count * 0.57)),
            ("TurboQuant INT4", int(param_count * 0.516)),
            ("TurboQuant INT2", int(param_count * 0.266)),
        ]

        for name, size_bytes in methods:
            size_gb = size_bytes / (1024**3)
            fits = "✅" if size_bytes < 3.5 * (1024**3) else "❌"
            table.add_row(name, f"{size_gb:.2f} GB", fits)

        console.print(table)
