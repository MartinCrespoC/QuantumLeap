# TurboQuant Deployment

API servers, scripts, Docker configs, and monitoring.

## Quick Start

### Launch a model (CLI)
```bash
bash scripts/launch_model.sh llama-3.2-3b
```

### Launch API server
```bash
bash scripts/launch_model.sh --api --model llama3.2:3b --port 8000
```

### Docker deployment
```bash
docker compose -f docker/docker-compose.yml up -d
```

## Scripts

- **`launch_model.sh`** — Intelligent launcher with auto GPU/CPU offloading
- **`optimize_system.sh`** — System-level optimizations for LLM inference
- **`benchmark.sh`** — Full benchmark suite

## Model Configs

Pre-configured YAML files in `configs/models/`:
- `llama-3b-turboquant.yaml` — Full GPU (0.8 GB VRAM)
- `mistral-7b-turboquant.yaml` — Full GPU (2.3 GB VRAM)
- `qwen-72b-turboquant.yaml` — GPU+CPU offload

## API

OpenAI-compatible REST API at `http://localhost:8000/v1/`:
- `GET /v1/models` — List models
- `POST /v1/chat/completions` — Chat completions
- `GET /health` — Health check
