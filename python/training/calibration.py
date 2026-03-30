"""Calibration data collection for post-training quantization."""

from __future__ import annotations

from typing import Optional

import torch
from rich.console import Console

console = Console()


def collect_calibration_data(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    max_length: int = 512,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Collect activation statistics for calibration.

    Args:
        model: Pre-trained model to calibrate.
        tokenizer: Tokenizer for the model.
        texts: Calibration text samples.
        max_length: Maximum sequence length.
        device: Target device.

    Returns:
        Dictionary mapping layer names to activation statistics.
    """
    stats: dict[str, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(name: str):
        def hook_fn(module, input, output):
            if name not in stats:
                stats[name] = {"min": [], "max": [], "absmax": []}
            if isinstance(output, torch.Tensor):
                stats[name]["min"].append(output.min().item())
                stats[name]["max"].append(output.max().item())
                stats[name]["absmax"].append(output.abs().max().item())
        return hook_fn

    # Register hooks on all linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))

    # Run calibration
    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(
                text, return_tensors="pt", max_length=max_length, truncation=True
            ).to(device)
            model(**inputs)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Aggregate statistics
    result: dict[str, torch.Tensor] = {}
    for name, stat in stats.items():
        result[f"{name}.range"] = torch.tensor(
            [min(stat["min"]), max(stat["max"])]
        )
        result[f"{name}.absmax"] = torch.tensor(max(stat["absmax"]))

    console.print(f"[green]Calibration complete: {len(stats)} layers profiled[/green]")
    return result
