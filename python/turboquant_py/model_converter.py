"""Model Converter: Convert HuggingFace/GGUF models to TurboQuant format."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from rich.console import Console
from rich.progress import Progress

from turboquant_py.quantizer import TurboQuantizer, QuantConfig, QuantBits

console = Console()


class ModelConverter:
    """Convert models between formats with TurboQuant quantization."""

    def __init__(self, config: Optional[QuantConfig] = None) -> None:
        self.config = config or QuantConfig()
        self.quantizer = TurboQuantizer(self.config)

    def convert_safetensors(
        self,
        input_path: str | Path,
        output_path: str | Path,
        bits: QuantBits = QuantBits.INT2,
    ) -> dict[str, float]:
        """Convert a safetensors model to TurboQuant format.

        Args:
            input_path: Path to input safetensors file.
            output_path: Path for output TurboQuant file.
            bits: Target quantization bits.

        Returns:
            Dictionary with conversion metrics.
        """
        from safetensors.torch import load_file, save_file

        input_path = Path(input_path)
        output_path = Path(output_path)

        console.print(f"[cyan]Loading model from {input_path}...[/cyan]")
        tensors = load_file(str(input_path))

        total_params = sum(t.numel() for t in tensors.values())
        console.print(f"[cyan]Total parameters: {total_params / 1e9:.2f}B[/cyan]")

        self.quantizer.estimate_model_size(total_params)

        quantized_tensors: dict[str, torch.Tensor] = {}
        total_mse = 0.0
        total_compression = 0.0
        count = 0

        with Progress() as progress:
            task = progress.add_task("Quantizing...", total=len(tensors))

            for name, tensor in tensors.items():
                if tensor.numel() < 256:
                    # Keep small tensors (biases, norms) in FP16
                    quantized_tensors[name] = tensor.half()
                else:
                    result = self.quantizer.quantize_tensor(tensor)
                    # Store as uint8 tensor
                    packed_tensor = torch.frombuffer(
                        result.packed_data, dtype=torch.uint8
                    )
                    quantized_tensors[name] = packed_tensor
                    quantized_tensors[f"{name}.scales"] = torch.from_numpy(
                        result.scales
                    )
                    quantized_tensors[f"{name}.zeros"] = torch.from_numpy(
                        result.zero_points
                    )

                    total_mse += result.mse
                    total_compression += result.compression_ratio
                    count += 1

                progress.update(task, advance=1)

        console.print(f"\n[green]Saving to {output_path}...[/green]")
        save_file(quantized_tensors, str(output_path))

        metrics = {
            "total_params": total_params,
            "avg_mse": total_mse / max(count, 1),
            "avg_compression": total_compression / max(count, 1),
            "input_size_mb": input_path.stat().st_size / (1024 * 1024),
            "output_size_mb": output_path.stat().st_size / (1024 * 1024),
        }

        console.print(f"[green]Conversion complete![/green]")
        console.print(f"  Input:  {metrics['input_size_mb']:.1f} MB")
        console.print(f"  Output: {metrics['output_size_mb']:.1f} MB")
        console.print(f"  Avg MSE: {metrics['avg_mse']:.6f}")
        console.print(
            f"  Compression: {metrics['avg_compression']:.1f}x"
        )

        return metrics

    def estimate_offload(
        self, model_path: str | Path, vram_mb: int = 3500
    ) -> dict[str, int]:
        """Estimate optimal GPU/CPU layer split.

        Args:
            model_path: Path to model config or safetensors.
            vram_mb: Available VRAM in MB.

        Returns:
            Dict with gpu_layers, cpu_layers, estimated tokens/s.
        """
        # TODO: Parse model config to get layer count and sizes
        console.print("[yellow]Offload estimation not yet implemented.[/yellow]")
        console.print("[yellow]Will be available after llama.cpp integration.[/yellow]")
        return {"gpu_layers": 0, "cpu_layers": 0, "estimated_tps": 0}
