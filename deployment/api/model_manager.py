"""Model Manager: Load, unload, and manage TurboQuant models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


@dataclass
class LoadedModel:
    """Represents a loaded model in memory."""

    name: str
    path: Path
    bits: int
    gpu_layers: int
    cpu_layers: int
    vram_usage_mb: float
    ram_usage_mb: float
    max_context: int


class ModelManager:
    """Manage multiple TurboQuant models."""

    def __init__(self, models_dir: Optional[str] = None) -> None:
        self.models_dir = Path(models_dir or "~/.cache/turboquant/models").expanduser()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.loaded: dict[str, LoadedModel] = {}

    def list_available(self) -> list[str]:
        """List available model configs."""
        configs_dir = Path(__file__).parent.parent / "configs" / "models"
        if configs_dir.exists():
            return [f.stem for f in configs_dir.glob("*.yaml")]
        return []

    def load_model(self, name: str, bits: int = 2) -> Optional[LoadedModel]:
        """Load a model into memory.

        Args:
            name: Model name (e.g., 'llama-3b-turboquant').
            bits: Quantization bits.

        Returns:
            LoadedModel instance or None if failed.
        """
        if name in self.loaded:
            console.print(f"[yellow]Model '{name}' already loaded.[/yellow]")
            return self.loaded[name]

        # TODO: Implement actual model loading with TurboQuant
        console.print(f"[cyan]Loading model '{name}' with INT{bits}...[/cyan]")
        console.print("[yellow]Model loading not yet implemented.[/yellow]")
        return None

    def unload_model(self, name: str) -> bool:
        """Unload a model from memory."""
        if name in self.loaded:
            del self.loaded[name]
            console.print(f"[green]Model '{name}' unloaded.[/green]")
            return True
        return False

    def get_status(self) -> dict:
        """Get status of all loaded models."""
        return {
            "models_loaded": len(self.loaded),
            "models": {
                name: {
                    "bits": m.bits,
                    "gpu_layers": m.gpu_layers,
                    "vram_mb": m.vram_usage_mb,
                }
                for name, m in self.loaded.items()
            },
        }
