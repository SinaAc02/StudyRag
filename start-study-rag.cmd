@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found.
  echo Create it and run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)

start "" "http://127.0.0.1:8000"
set "HF_HOME=%~dp0data\model_cache"
echo Study RAG is starting at http://127.0.0.1:8000
echo PDFs are loaded automatically from the documents folder.
echo Press Ctrl+C to stop.
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
