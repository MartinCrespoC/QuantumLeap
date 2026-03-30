@echo off
REM QuantumLeap Quick Start — Windows
REM 801% faster LLM inference built on llama.cpp
REM Requires: Python 3.10+, CUDA Toolkit (optional for GPU)

set SCRIPT_DIR=%~dp0
set PROJECT_ROOT=%SCRIPT_DIR%..

cd /d "%PROJECT_ROOT%"

echo.
echo   ⚛️  QuantumLeap v0.4.0 — 801% Faster LLM Inference
echo   ══════════════════════════════════════════════════
echo   Built on llama.cpp ^| Powered by TurboQuant Engine
echo.

REM Check engine
if not exist "engine\llama.cpp\build\bin\llama-server.exe" (
    if not exist "engine\llama.cpp\build\bin\Release\llama-server.exe" (
        echo   ❌ Engine not built. Run: scripts\setup.bat
        exit /b 1
    )
)

REM Detect GPU
where nvidia-smi >nul 2>&1
if %ERRORLEVEL% equ 0 (
    for /f "tokens=*" %%a in ('nvidia-smi --query-gpu^=name --format^=csv^,noheader 2^>nul') do echo   GPU: %%a
) else (
    echo   GPU: Not detected (CPU-only mode^)
)

REM Python venv
if not exist ".venv" if not exist "venv" (
    echo   📦 Creating Python virtual environment...
    python -m venv .venv
)

if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

python -c "import fastapi" 2>nul
if %ERRORLEVEL% neq 0 (
    echo   📦 Installing Python dependencies...
    pip install -q -r api\requirements.txt
)

echo.
echo   ✅ Starting QuantumLeap Server...
echo.
echo   🌐 Web UI:     http://localhost:11434
echo   🔌 Ollama API: http://localhost:11434/api/
echo   🤖 OpenAI API: http://localhost:11434/v1/
echo   📊 Features:   Auto-offloading, UMA, Requantization, Smart Search
echo.
echo   Press Ctrl+C to stop
echo.

python api\server.py %*
