"""Quantization-Aware Training (QAT) for TurboQuant models."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class TurboQuantSTE(torch.autograd.Function):
    """Straight-Through Estimator for TurboQuant quantization.

    Allows gradient flow through quantization during training.
    Forward: quantize to INT2/INT4
    Backward: pass gradients straight through (identity)
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, bits: int, group_size: int
    ) -> torch.Tensor:
        max_quant = (1 << bits) - 1
        n = x.numel()
        flat = x.view(-1)

        # Group-wise quantize
        num_groups = (n + group_size - 1) // group_size
        output = torch.zeros_like(flat)

        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, n)
            group = flat[start:end]

            min_val = group.min()
            max_val = group.max()
            range_val = max_val - min_val

            if range_val < 1e-8:
                output[start:end] = group
                continue

            scale = range_val / max_quant
            q = torch.round((group - min_val) / scale).clamp(0, max_quant)
            output[start:end] = q * scale + min_val

        return output.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        return grad_output, None, None


def quantize_ste(
    x: torch.Tensor, bits: int = 2, group_size: int = 128
) -> torch.Tensor:
    """Apply TurboQuant quantization with STE for training."""
    return TurboQuantSTE.apply(x, bits, group_size)


class QuantizedLinear(nn.Module):
    """Linear layer with TurboQuant quantization during forward pass."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 2,
        group_size: int = 128,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w_quant = quantize_ste(self.weight, self.bits, self.group_size)
        else:
            w_quant = self.weight
        return nn.functional.linear(x, w_quant, self.bias)


def replace_linear_with_quantized(
    model: nn.Module,
    bits: int = 2,
    group_size: int = 128,
    skip_patterns: Optional[list[str]] = None,
) -> nn.Module:
    """Replace all nn.Linear layers with QuantizedLinear.

    Args:
        model: PyTorch model to modify.
        bits: Quantization bits (2 or 4).
        group_size: Elements per quantization group.
        skip_patterns: Layer name patterns to skip (e.g., ["lm_head", "embed"]).

    Returns:
        Modified model with quantized linear layers.
    """
    skip_patterns = skip_patterns or ["lm_head", "embed", "norm"]

    for name, module in model.named_modules():
        if any(pat in name for pat in skip_patterns):
            continue

        if isinstance(module, nn.Linear):
            quantized = QuantizedLinear(
                module.in_features,
                module.out_features,
                bits=bits,
                group_size=group_size,
                bias=module.bias is not None,
            )
            quantized.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                quantized.bias.data.copy_(module.bias.data)

            # Replace in parent
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], quantized)
            else:
                setattr(model, name, quantized)

    return model
