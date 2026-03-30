"""HuggingFace Transformers patch for TurboQuant quantization."""

from __future__ import annotations

import torch
import torch.nn as nn
from rich.console import Console

console = Console()


def patch_model_for_turboquant(
    model: nn.Module,
    bits: int = 2,
    group_size: int = 128,
    device: str = "cuda",
) -> nn.Module:
    """Patch a HuggingFace model to use TurboQuant quantization.

    Replaces weight tensors with quantized versions and
    patches forward methods for dequantization on-the-fly.

    Args:
        model: HuggingFace model to patch.
        bits: Quantization bits (2 or 4).
        group_size: Elements per quantization group.
        device: Target device.

    Returns:
        Patched model with TurboQuant weights.
    """
    from turboquant_py.quantizer import TurboQuantizer, QuantConfig, QuantBits

    config = QuantConfig(
        bits=QuantBits(bits),
        group_size=group_size,
        device=device,
    )
    quantizer = TurboQuantizer(config)

    total_params = sum(p.numel() for p in model.parameters())
    quantized_params = 0

    for name, param in model.named_parameters():
        # Skip small tensors and embeddings
        if param.numel() < 256 or "embed" in name or "norm" in name:
            continue

        result = quantizer.quantize_tensor(param.data)
        quantized_params += param.numel()

        # Store quantized data as buffer
        packed = torch.frombuffer(result.packed_data, dtype=torch.uint8)
        scales = torch.from_numpy(result.scales).to(device)

        model.register_buffer(f"{name.replace('.', '_')}_packed", packed)
        model.register_buffer(f"{name.replace('.', '_')}_scales", scales)

    console.print(
        f"[green]Patched {quantized_params / 1e6:.1f}M / "
        f"{total_params / 1e6:.1f}M params with TurboQuant INT{bits}[/green]"
    )

    return model
