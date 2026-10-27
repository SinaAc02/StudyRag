from pathlib import Path
import os
import sys


def application_home() -> Path:
    """Use the checkout in source runs and user-owned storage in installed builds."""
    if override := os.environ.get("STUDYRAG_HOME"):
        return Path(override).expanduser().resolve()
    checkout = Path(__file__).resolve().parents[2]
    if (checkout / "pyproject.toml").is_file() and (checkout / "src/studyrag").is_dir():
        return checkout
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    return (base / "StudyRag").resolve()


PACKAGE_IMPORT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = application_home()
DEFAULT_DATABASE = PROJECT_ROOT / "data/studyrag.sqlite"
DEFAULT_MATERIAL_FOLDER = PROJECT_ROOT / "material"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
DEFAULT_GENERATION_MODEL = "qwen3.5:9b"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"


def worker_python() -> str:
    """Use console Python with redirected pipes even when the GUI uses pythonw."""
    executable = Path(sys.executable)
    if executable.name.casefold() == "pythonw.exe":
        return str(executable.with_name("python.exe"))
    return str(executable)
