"""TurboQuant Speed Benchmarks."""

from __future__ import annotations

import time

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from turboquant_py.quantizer import TurboQuantizer, QuantConfig, QuantBits

console = Console()


def benchmark_quantize(sizes: list[int], bits: QuantBits, iterations: int = 10):
    """Benchmark quantization speed for different tensor sizes."""
    config = QuantConfig(bits=bits, group_size=128)
    quantizer = TurboQuantizer(config)

    table = Table(title=f"Quantization Speed (INT{int(bits)})")
    table.add_column("Size", style="cyan")
    table.add_column("Time (ms)", style="green")
    table.add_column("Throughput (M elem/s)", style="yellow")
    table.add_column("Compression", style="magenta")

    for size in sizes:
        tensor = torch.randn(size)

        # Warmup
        quantizer.quantize_tensor(tensor)

        # Benchmark
        start = time.perf_counter()
        for _ in range(iterations):
            result = quantizer.quantize_tensor(tensor)
        elapsed = (time.perf_counter() - start) / iterations * 1000

        throughput = size / (elapsed / 1000) / 1e6

        table.add_row(
            f"{size:,}",
            f"{elapsed:.2f}",
            f"{throughput:.1f}",
            f"{result.compression_ratio:.1f}x",
        )

    console.print(table)


def benchmark_model_sizes():
    """Estimate quantization impact on common model sizes."""
    table = Table(title="Model Size Estimates (TurboQuant)")
    table.add_column("Model", style="cyan")
    table.add_column("Params", style="white")
    table.add_column("FP16", style="red")
    table.add_column("Q4_K_M", style="yellow")
    table.add_column("TQ4", style="green")
    table.add_column("TQ2", style="green", justify="right")
    table.add_column("Fits 4GB?", style="magenta")

    models = [
        ("Llama 3.2 1B", 1.2e9),
        ("Llama 3.2 3B", 3.2e9),
        ("Mistral 7B", 7.3e9),
        ("Llama 3.1 8B", 8.0e9),
        ("Llama 3.1 13B", 13.0e9),
        ("Qwen 2.5 32B", 32.5e9),
        ("Llama 3.1 70B", 70.6e9),
    ]

    for name, params in models:
        fp16 = params * 2 / (1024**3)
        q4km = params * 0.57 / (1024**3)
        tq4 = params * 0.516 / (1024**3)
        tq2 = params * 0.266 / (1024**3)
        fits = "✅" if tq2 < 3.5 else "❌"

        table.add_row(
            name,
            f"{params / 1e9:.1f}B",
            f"{fp16:.1f} GB",
            f"{q4km:.1f} GB",
            f"{tq4:.1f} GB",
            f"{tq2:.1f} GB",
            fits,
        )

    console.print(table)


def main():
    console.print("[bold cyan]=== TurboQuant Python Benchmarks ===[/bold cyan]\n")

    # Hardware info
    quantizer = TurboQuantizer()
    quantizer.print_hardware_info()
    console.print()

    # Model size estimates
    benchmark_model_sizes()
    console.print()

    # Quantization speed
    sizes = [1024, 65536, 1_048_576, 16_777_216]

    benchmark_quantize(sizes, QuantBits.INT2, iterations=5)
    console.print()

    benchmark_quantize(sizes, QuantBits.INT4, iterations=5)

    console.print("\n[bold green]=== Benchmarks Complete ===[/bold green]")


if __name__ == "__main__":
    main()
