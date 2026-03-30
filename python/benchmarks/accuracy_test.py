"""TurboQuant Accuracy Tests (Python)."""

from __future__ import annotations

import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from turboquant_py.quantizer import TurboQuantizer, QuantConfig, QuantBits

console = Console()


def test_quantize_accuracy(bits: QuantBits, sizes: list[int]):
    """Test quantization accuracy for various tensor sizes and distributions."""
    config = QuantConfig(bits=bits, group_size=128)
    quantizer = TurboQuantizer(config)

    table = Table(title=f"Accuracy Test: INT{int(bits)}")
    table.add_column("Distribution", style="cyan")
    table.add_column("Size", style="white")
    table.add_column("MSE", style="green")
    table.add_column("Max Error", style="yellow")
    table.add_column("PASS?", style="magenta")

    distributions = [
        ("Normal(0,1)", lambda n: torch.randn(n)),
        ("Uniform(-1,1)", lambda n: torch.rand(n) * 2 - 1),
        ("Normal(0,0.1)", lambda n: torch.randn(n) * 0.1),
        ("Sparse (90%)", lambda n: torch.randn(n) * (torch.rand(n) > 0.9).float()),
    ]

    all_pass = True
    for dist_name, gen_fn in distributions:
        for size in sizes:
            tensor = gen_fn(size)
            result = quantizer.quantize_tensor(tensor)

            # Threshold depends on bit width
            mse_threshold = 0.1 if bits == QuantBits.INT2 else 0.01
            passed = result.mse < mse_threshold

            if not passed:
                all_pass = False

            table.add_row(
                dist_name,
                f"{size:,}",
                f"{result.mse:.6f}",
                f"{result.max_error:.6f}",
                "✅" if passed else "❌",
            )

    console.print(table)
    return all_pass


def main():
    console.print("[bold cyan]=== TurboQuant Accuracy Tests ===[/bold cyan]\n")

    sizes = [256, 4096, 65536]

    pass_int2 = test_quantize_accuracy(QuantBits.INT2, sizes)
    console.print()

    pass_int4 = test_quantize_accuracy(QuantBits.INT4, sizes)

    console.print()
    if pass_int2 and pass_int4:
        console.print("[bold green]All accuracy tests PASSED ✅[/bold green]")
    else:
        console.print("[bold red]Some accuracy tests FAILED ❌[/bold red]")
        exit(1)


if __name__ == "__main__":
    main()
