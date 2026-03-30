# TurboQuant Python

Python bindings, model conversion, and training tools.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Usage

### Quantize a tensor
```python
from turboquant_py import TurboQuantizer, QuantConfig, QuantBits
import torch

config = QuantConfig(bits=QuantBits.INT2, group_size=128)
quantizer = TurboQuantizer(config)
quantizer.print_hardware_info()

tensor = torch.randn(4096, 4096)
result = quantizer.quantize_tensor(tensor)
print(f"MSE: {result.mse:.6f}, Compression: {result.compression_ratio:.1f}x")
```

### Convert a model
```python
from turboquant_py import ModelConverter
converter = ModelConverter()
converter.convert_safetensors("model.safetensors", "model-tq2.safetensors")
```

## Benchmarks

```bash
python benchmarks/speed_test.py     # Speed benchmarks
python benchmarks/accuracy_test.py  # Accuracy tests
```
