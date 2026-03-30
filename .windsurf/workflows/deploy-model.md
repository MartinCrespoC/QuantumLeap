---
description: Deploy a quantized LLM model with API server and Web UI
---

# Deploy Model Workflow

1. Ensure Ollama is installed
```bash
which ollama || curl -fsSL https://ollama.ai/install.sh | sh
```

2. Pull the target model
```bash
ollama pull llama3.2:3b
```

3. Run system optimizations
```bash
sudo bash ~/Documentos/Proyectos/llm-turbo/deployment/scripts/optimize_system.sh
```

4. Start API server (background)
```bash
bash ~/Documentos/Proyectos/llm-turbo/deployment/scripts/launch_model.sh --api --model llama3.2:3b --port 8000
```

5. Verify API health
// turbo
```bash
curl -s http://localhost:8000/api/tags | python3 -m json.tool
```

6. Test generation
```bash
curl -s http://localhost:8000/api/generate -d '{"model":"llama3.2:3b","prompt":"Hello, how are you?","stream":false}' | python3 -m json.tool
```

7. (Optional) Start Open WebUI
```bash
pip install open-webui && open-webui serve --port 3000
```
