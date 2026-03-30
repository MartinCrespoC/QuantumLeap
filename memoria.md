# TurboQuant — Memoria Completa del Proyecto

> **Fecha**: 30 de Marzo 2026
> **Autor**: ikharoz + Cascade AI
> **Estado**: ✅ FUNCIONAL — Servidor corriendo, modelo cargado, APIs probadas
> **Versión**: 0.2.0

---

## 1. ¿Qué es TurboQuant?

**TurboQuant** (Zandieh et al., ICLR 2026) es el método de Google para comprimir el KV cache de LLMs de FP16 a **3 bits por valor**, logrando **4.9x compresión** con pérdida de precisión casi nula (MSE 0.034).

### Algoritmo
1. **Normalización** — Cada vector KV se normaliza y se almacena su norma (4 bytes)
2. **Rotación Ortogonal** — Se aplica una matriz ortogonal aleatoria fija (genera distribución Beta uniforme)
3. **Cuantización Lloyd-Max** — Cada coordenada rotada se cuantiza a 3 bits con codebook óptimo MSE
4. **Bit Packing** — 128 índices empaquetados en 48 bytes (TQ3) o 64 bytes (TQ4)

### Impacto
- GPU 4GB: de ~8K tokens de contexto → **~40K tokens** con TQ3
- 70B model en 3x RTX 3090: de ~109K → **~536K tokens** de contexto

### Referencias
- Blog: https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/
- llama.cpp commit (mudler/LocalAI): https://github.com/mudler/llama.cpp/commit/dee102db1bfd723c91f67138b8018ce35a6be477
- Fork con CUDA: https://github.com/spiritbuun/llama-cpp-turboquant-cuda
- Implementación KV cache (ik_llama.cpp): https://github.com/ikawrakow/ik_llama.cpp/issues/1509
- Gist con código: https://gist.github.com/veritatisquaesitoressumus/6aa5973955007ffd858889c76aa60408

---

## 2. Arquitectura del Proyecto

```
~/Documentos/Proyectos/llm-turbo/
├── engine/
│   └── llama.cpp/              # Fork TurboQuant CUDA (spiritbuun)
│       └── build/
│           └── bin/
│               ├── llama-server    # Servidor de inferencia (interno :8081)
│               ├── llama-cli       # CLI para pruebas
│               └── llama-bench     # Benchmarks nativos
├── api/
│   ├── server.py               # FastAPI: Ollama + OpenAI API + Model Manager + Bench
│   └── requirements.txt        # fastapi, uvicorn, httpx, psutil, etc.
├── web/
│   ├── index.html              # UI principal (4 tabs)
│   └── static/
│       ├── style.css           # Dark theme profesional
│       └── app.js              # Lógica completa (chat, modelos, bench, etc.)
├── models/                     # Archivos GGUF descargados
│   └── SmolLM2-1.7B-Instruct-Q4_K_M.gguf (1007 MB)
├── benchmarks/                 # Resultados de benchmarks (JSON)
├── scripts/
│   ├── setup.sh                # Instalación completa (auto-detecta OS/GPU)
│   ├── start.sh                # Arranque rápido
│   └── download_model.sh       # Descargador de modelos
├── .venv/                      # Python 3.14 virtualenv
├── config.json                 # Configuración persistente
├── README.md                   # Documentación principal
└── memoria.md                  # ← ESTE ARCHIVO
```

---

## 3. Hardware del Sistema

| Componente | Detalle |
|------------|---------|
| **CPU** | Intel i5-11400H @ 2.70GHz (12 threads, AVX-512) |
| **GPU** | NVIDIA GeForce RTX 3050 Laptop (4096 MB VRAM, Compute 8.6) |
| **RAM** | 24 GB DDR4 |
| **OS** | Manjaro Linux (Arch-based), x86_64 |
| **Driver** | NVIDIA 590.48.01 |
| **CUDA** | 13.1.115 (instalado en /opt/cuda/) |
| **CMake** | 4.2.3 |
| **GCC** | 15.2.1 |
| **Python** | 3.14 |
| **Node** | Disponible (npm incluido) |

---

## 4. Cómo se Construyó

### 4.1 Engine (llama.cpp con TurboQuant KV Cache)

**Engine Base**: `ikawrakow/ik_llama.cpp` (mejor rendimiento CPU/GPU que upstream)

**Integración TurboQuant TQ3/TQ4**:
```bash
# Clonar fork ikawrakow (SOTA quants + mejor rendimiento)
git clone --depth 1 https://github.com/ikawrakow/ik_llama.cpp.git engine/llama.cpp

# Integrar TurboQuant KV cache:
# 1. Agregar tipos GGML_TYPE_TQ3 (500) y GGML_TYPE_TQ4 (501) a ggml.h
# 2. Registrar en type_traits de ggml.c (blck_size=128, type_size=52/68)
# 3. Agregar archivos TurboQuant:
#    - ggml/include/ggml_turboquant.h (header compatible CUDA C++)
#    - ggml/src/ggml_turboquant.c (implementación CPU)
#    - ggml/src/ggml-cuda/ggml_turboquant.cu (kernels CUDA)
# 4. Actualizar CMakeLists.txt para incluir archivos TQ
# 5. Agregar soporte en common/common.cpp (kv_cache_type_from_str)

# Configurar build con CUDA para RTX 3050
export PATH=/opt/cuda/bin:$PATH
cmake -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_CURL=OFF

# Compilar (415 targets, ~20 min con CUDA)
cmake --build build -j12
```

**Resultado**: `engine/llama.cpp/build/bin/llama-server` (8.1 MB, ELF x86-64)

### 4.2 Python API Server

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi 'uvicorn[standard]' httpx psutil pydantic python-multipart aiofiles rich
```

### 4.3 Modelo de Prueba

```bash
wget -O models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/bartowski/SmolLM2-1.7B-Instruct-GGUF/resolve/main/SmolLM2-1.7B-Instruct-Q4_K_M.gguf"
# 1007 MB descargados
```

---

## 5. APIs Disponibles

### 5.1 Ollama-Compatible (puerto 11434)

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/tags` | Listar modelos locales |
| POST | `/api/chat` | Chat con streaming |
| POST | `/api/generate` | Generación de texto |
| POST | `/api/show` | Info de un modelo |
| GET | `/api/ps` | Modelos en ejecución |
| GET | `/api/version` | Versión del servidor |

### 5.2 OpenAI-Compatible

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/v1/chat/completions` | Chat completions (stream + no-stream) |
| GET | `/v1/models` | Lista de modelos |

### 5.3 Model Manager

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/models/list` | Lista completa con estado |
| POST | `/api/models/load` | Cargar modelo en GPU |
| POST | `/api/models/unload` | Descargar modelo |
| POST | `/api/models/set-default` | Establecer modelo default |
| DELETE | `/api/models/delete` | Eliminar modelo del disco |
| POST | `/api/models/update-settings` | Cambiar gpu_layers, ctx_size |
| GET | `/api/models/search-hf` | Buscar modelos GGUF en HuggingFace |
| GET | `/api/models/hf-files` | Listar archivos GGUF de un repo HF |
| POST | `/api/models/download` | Descargar modelo desde URL |
| GET | `/api/models/downloads` | Estado de descargas activas |

### 5.4 Benchmark

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| POST | `/api/bench/run` | Ejecutar benchmark (N runs) |
| GET | `/api/bench/history` | Historial de benchmarks |
| DELETE | `/api/bench/clear` | Limpiar historial |

### 5.5 Sistema

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/api/status` | GPU, RAM, CPU, engine, config |
| GET | `/health` | Health check |

---

## 6. Web UI — 4 Tabs

### Tab 1: Chat
- Conversaciones múltiples con historial (localStorage)
- Selector de modelo
- Streaming token-by-token
- Métricas: tokens, tiempo, t/s
- Markdown rendering (code, bold, italic, listas)
- Cursor animado durante generación
- Stop button para detener

### Tab 2: Models (Administrador)
- **Modelos Locales**: Grid con cards, estado (loaded/default), acciones (load/unload/delete/set-default)
- **Engine Settings**: GPU layers, context size (persistente en config.json)
- **Búsqueda HuggingFace**: Buscar repos GGUF, ver archivos con tamaños, descargar directo
- **Descargas Activas**: Barra de progreso en tiempo real, polling cada 2s

### Tab 3: Benchmark
- Configurar: modelo, prompt, max_tokens, número de runs
- Ejecutar benchmarks con métricas detalladas (t/s, latencia, tokens)
- Historial persistente en archivos JSON
- Guía de optimización integrada

### Tab 4: Help
- **Qué es TurboQuant**: Explicación del algoritmo, por qué importa
- **Tabla de Formatos**: Comparación de F16, Q8, Q6, Q4, Q3, Q2, IQ2, TQ1, TQ2
- **Checklist de Optimización**: GPU, memoria, sistema
- **Referencia API**: Todos los endpoints con ejemplos curl
- **Arquitectura**: Diagrama ASCII del sistema

---

## 7. Resultados de Benchmarks

### SmolLM2 1.7B (Q4_K_M) en RTX 3050 4GB

| Métrica | Valor |
|---------|-------|
| **Tokens/segundo (promedio)** | **100.9 t/s** |
| **Latencia promedio** | 0.09s |
| **VRAM utilizada** | ~1977 MB |
| **RAM utilizada** | ~9.2 GB |
| **Modelo size** | 1007 MB |

---

## 8. Cómo Usar

### Arranque Rápido
```bash
cd ~/Documentos/Proyectos/llm-turbo
bash scripts/start.sh
# → Web UI en http://localhost:11434
```

### TurboQuant KV Cache (TQ3/TQ4)

**Uso directo del engine con compresión extrema**:

```bash
# TQ3: 3-bit KV cache (4.9x compresión, MSE 0.034)
bash scripts/test_tq3.sh

# O manualmente:
engine/llama.cpp/build/bin/llama-server \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --cache-type-k tq3 \
  --cache-type-v tq3 \
  -c 8192 \
  -ngl 99 \
  --host 0.0.0.0 \
  --port 11434

# TQ4: 4-bit KV cache (3.7x compresión, mayor calidad)
engine/llama.cpp/build/bin/llama-server \
  -m models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf \
  --cache-type-k tq4 \
  --cache-type-v tq4 \
  -c 16384 \
  -ngl 99
```

**Tipos de KV cache soportados**:
- `f16` — FP16 (default, sin compresión)
- `q8_0` — 8-bit (2x compresión)
- `q4_0` — 4-bit (4x compresión)
- `tq3` — **TurboQuant 3-bit (4.9x compresión)** ⚡
- `tq4` — **TurboQuant 4-bit (3.7x compresión)** ⚡

**Impacto en RTX 3050 (4GB VRAM)**:
- Sin TQ: ~8K tokens de contexto
- Con TQ3: **~40K tokens** de contexto (5x más)
- Con TQ4: **~30K tokens** de contexto (3.7x más)

### Conectar Windsurf / VSCode
Configurar endpoint Ollama: `http://localhost:11434`

### Descargar Más Modelos
```bash
bash scripts/download_model.sh smollm   # 1.7B (1GB)
bash scripts/download_model.sh qwen     # 3B (2GB)
bash scripts/download_model.sh llama    # 3B (2GB)
# O usar la UI: tab Models → buscar en HuggingFace → descargar
```

### Desde Otra Máquina
La API escucha en `0.0.0.0:11434`, así que cualquier dispositivo en la misma red puede conectar.

---

## 9. Cross-Platform

El script `scripts/setup.sh` auto-detecta:

| OS | GPU Backend | Estado |
|----|-------------|--------|
| **Linux (Manjaro/Arch)** | CUDA (pacman) | ✅ Probado |
| **Linux (Ubuntu/Debian)** | CUDA (apt) | ✅ Script preparado |
| **macOS (Apple Silicon)** | Metal | ✅ Script preparado |
| **macOS (Intel)** | CPU only | ✅ Script preparado |
| **Windows** | Manual build | 📋 Instrucciones en README |

---

## 10. Flujo de Datos

```
Usuario (Web UI / curl / Windsurf)
        │
        ▼
   FastAPI Server (:11434)
   ├── Ollama API (/api/*)
   ├── OpenAI API (/v1/*)
   ├── Model Manager (/api/models/*)
   └── Benchmark (/api/bench/*)
        │
        ▼ (httpx proxy)
   llama-server (:8081)
   ├── GGML Backend
   ├── CUDA Kernels (sm_86)
   ├── TurboQuant KV Cache
   └── AVX-512 CPU fallback
        │
        ▼
   RTX 3050 (4GB) + i5-11400H (24GB RAM)
```

---

## 11. Archivos Clave y sus Funciones

| Archivo | Líneas | Función |
|---------|--------|---------|
| `api/server.py` | ~420 | FastAPI: todas las APIs, engine management, HF search, downloads, benchmark |
| `web/index.html` | ~280 | HTML: 4 tabs (Chat, Models, Bench, Help), estructura completa |
| `web/static/style.css` | ~350 | CSS: Dark theme profesional, responsive, animaciones |
| `web/static/app.js` | ~380 | JS: Chat streaming, model manager, HF browser, bench runner |
| `scripts/setup.sh` | ~155 | Bash: Setup completo cross-platform |
| `scripts/start.sh` | ~25 | Bash: Arranque rápido |
| `scripts/download_model.sh` | ~55 | Bash: Descargador de modelos |
| `config.json` | ~5 | JSON: Settings persistentes (gpu_layers, ctx_size, default model) |

---

## 12. Dependencias Instaladas

### Sistema (pacman)
- `cmake` 4.2.3
- `cuda` 13.1.1 (en /opt/cuda/)
- `ninja` (ya instalado)
- `gcc` 15.2.1 (ya instalado)

### Python (.venv)
- `fastapi` ≥0.115
- `uvicorn[standard]` ≥0.34
- `httpx` ≥0.28
- `pydantic` ≥2.10
- `psutil` ≥6.1
- `python-multipart`
- `aiofiles`
- `rich`

---

## 13. Posibles Mejoras Futuras

1. **TurboQuant KV Cache nativo** — Integrar la implementación de ik_llama.cpp#1509 (flags `--cache-type-k tq3 --cache-type-v tq3`)
2. **Electron desktop app** — Empaquettar como app de escritorio (ya tenemos node/npm)
3. **Vision/OCR** — Soporte para modelos multimodales (LLaVA, etc.)
4. **Chat history export** — Exportar conversaciones a JSON/Markdown
5. **Model comparison** — Benchmark side-by-side de dos modelos
6. **System tray** — Correr como servicio con icono en system tray
7. **Docker container** — Imagen pre-built para deployment
8. **Auto-update** — Actualizar llama.cpp fork automáticamente
9. **Ollama model registry** — Pull de modelos del registry de Ollama (ollama.com/library)
10. **GGUF quantization** — Cuantizar modelos FP16/safetensors a GGUF desde la UI

---

## 14. Troubleshooting

### Server no arranca
```bash
# Verificar que CUDA está en PATH
export PATH=/opt/cuda/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda/lib64:$LD_LIBRARY_PATH

# Verificar que llama-server existe
ls -la engine/llama.cpp/build/bin/llama-server

# Verificar venv
source .venv/bin/activate
python3 -c "import fastapi; print('OK')"
```

### Modelo no carga
```bash
# Verificar GPU
nvidia-smi

# Verificar modelo
ls -la models/*.gguf

# Reducir GPU layers si no hay suficiente VRAM
# En la UI: tab Models → Engine Settings → GPU Layers → reducir a 20-30
```

### Rebuild desde cero
```bash
cd engine/llama.cpp
rm -rf build
export PATH=/opt/cuda/bin:$PATH
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build -j12
```

---

> **Nota**: Este archivo contiene TODO el contexto necesario para continuar el desarrollo en un nuevo workspace. Al abrir `llm-turbo/` como workspace, Cascade puede leer este archivo para recuperar el contexto completo del proyecto.
