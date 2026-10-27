@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
    echo [ERROR] The project Python environment is missing. Follow README.md setup.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -B -c "from PySide6.QtWidgets import QApplication" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Desktop UI dependencies are missing. Install the project requirements.
    pause
    exit /b 1
)
if /I "%~1"=="--check" (
    echo [OK] Python and Qt are available. No app or model was started.
    exit /b 0
)
set "PYTHONPATH=%~dp0src"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
set "HF_HUB_OFFLINE=1"
set "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True"
if not exist "data" mkdir "data"
start "" ".venv\Scripts\pythonw.exe" -B -m studyrag desktop >>"data\desktop.log" 2>&1
exit /b 0
