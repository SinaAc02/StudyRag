from __future__ import annotations

import os
import json
import re
import sqlite3
import subprocess
import sys
import threading
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pymupdf

from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.ocr.paddle_structure import PaddleStructureEngine
from studyrag.ingestion.pipeline import processing_key_for
from studyrag.ingestion.store import INGESTION_SCHEMA_VERSION
from studyrag.settings import (
    DEFAULT_EMBEDDING_MODEL as EMBEDDING_MODEL,
    OLLAMA_BASE_URL,
    PACKAGE_IMPORT_ROOT,
    worker_python,
)


OCR_ENGINE = PaddleStructureEngine.name


def _invalid_document_processing_count(connection: sqlite3.Connection) -> int:
    expected_key = processing_key_for(PaddleStructureEngine(), ("fas", "eng", "deu"))
    return connection.execute(
        "SELECT COUNT(*) FROM documents d LEFT JOIN document_processing p "
        "ON p.document_hash = d.content_hash "
        "WHERE p.processing_key IS NULL OR p.processing_key != ?",
        (expected_key,),
    ).fetchone()[0]


def _invalid_page_count(connection: sqlite3.Connection) -> int:
    """Count pages with incomplete or incompatible extraction metadata."""
    return connection.execute(
        "SELECT COUNT(*) FROM pages p WHERE COALESCE("
        "(p.text_source = 'ocr' AND p.ocr_engine = ?) OR "
        "(p.text_source = 'native' AND p.ocr_engine IS NULL "
        "AND LENGTH(TRIM(p.normalized_text)) >= 120 "
        "AND EXISTS (SELECT 1 FROM page_blocks b WHERE "
        "b.document_hash = p.document_hash AND b.page_number = p.page_number)), 0) = 0",
        (OCR_ENGINE,),
    ).fetchone()[0]


class PreparationCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class PreparationStatus:
    stage: str = "checking"
    message: str = "Checking the library..."
    percent: int = 0
    ready: bool = False
    running: bool = False
    error: str | None = None
    current: int = 0
    total: int = 0


class PreparationCoordinator:
    def __init__(self, project_root: Path, material_folder: Path, database_path: Path) -> None:
        self.project_root = project_root.resolve()
        self.material_folder = material_folder.resolve()
        self.database_path = database_path.resolve()
        self.staging_path = self.database_path.with_name("studyrag.building.sqlite")
        self.backup_path = self.database_path.with_name("studyrag.previous.sqlite")
        self._status = PreparationStatus()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel_requested = threading.Event()
        self._process_lock = threading.Lock()
        self._process: subprocess.Popen[str] | None = None

    def status(self) -> dict[str, object]:
        with self._lock:
            return asdict(self._status)

    def start(self, material_folder: Path | None = None, *, force: bool = False) -> bool:
        with self._lock:
            if self._status.running:
                return False
            if material_folder is not None:
                self.material_folder = material_folder.resolve()
            self._cancel_requested.clear()
            self._status = PreparationStatus(running=True)
            self._thread = threading.Thread(
                target=self._run,
                kwargs={"force": force},
                daemon=True,
                name="studyrag-preparation",
            )
            self._thread.start()
            return True

    def cancel(self) -> None:
        self._cancel_requested.set()
        with self._process_lock:
            process = self._process
        if process is not None and process.poll() is None:
            print("[StudyRag] Stopping the active preparation worker...", flush=True)
            process.terminate()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=3)
        if process is not None and process.poll() is None:
            process.kill()

    def _check_cancelled(self) -> None:
        if self._cancel_requested.is_set():
            raise PreparationCancelled("Preparation stopped because the application was closed")

    def _set(self, **changes: object) -> None:
        with self._lock:
            for key, value in changes.items():
                setattr(self._status, key, value)
            message = changes.get("message")
            if message:
                print(f"[StudyRag] {message}", flush=True)

    def _run(self, *, force: bool) -> None:
        try:
            self._set(stage="checking", message="Checking documents and index...", percent=1)
            total_pages = self._page_count()
            self._set(total=total_pages)
            self._check_cancelled()
            if not force and self._is_fresh():
                self._set(
                    stage="ready",
                    message="Library is current and ready.",
                    percent=100,
                    ready=True,
                    running=False,
                )
                return

            if not force and self._database_is_complete(self.staging_path):
                self._set(
                    stage="activating",
                    message="Recovering the completed library from staging...",
                    percent=99,
                )
                self._activate_staging()
                self._set(
                    stage="ready",
                    message="Library is ready. You can study now.",
                    percent=100,
                    ready=True,
                    running=False,
                    current=total_pages,
                )
                return

            self._set(stage="unloading", message="Unloading the Ollama study model...", percent=3)
            self._unload_ollama()
            self._check_cancelled()
            self._prepare_staging(force=force)

            self._set(
                stage="ocr",
                message=("Starting native extraction and selective PP-StructureV3 in an isolated process..."),
                percent=5,
            )
            self._run_command(
                [
                    worker_python(),
                    "-m",
                    "studyrag",
                    "ingest",
                    str(self.material_folder),
                    "--database",
                    str(self.staging_path),
                ],
                total_pages=total_pages,
            )
            self._check_cancelled()

            self._set(
                stage="releasing_vram",
                message="PP-StructureV3 process exited; GPU memory released.",
                percent=82,
            )
            self._set(stage="indexing", message="Building the multilingual index...", percent=84)
            self._run_command(
                [
                    worker_python(),
                    "-m",
                    "studyrag",
                    "index",
                    "--database",
                    str(self.staging_path),
                ]
            )
            self._check_cancelled()
            self._validate_staging()
            self._set(stage="activating", message="Activating the completed library...", percent=99)
            self._activate_staging()
            self._set(
                stage="ready",
                message="Library is ready. You can study now.",
                percent=100,
                ready=True,
                running=False,
                current=total_pages,
            )
        except PreparationCancelled as error:
            self._set(
                stage="stopped",
                message=str(error),
                running=False,
                ready=False,
                error=None,
            )
        except Exception as error:
            self._set(
                stage="error",
                message=f"Preparation failed: {error}",
                running=False,
                ready=False,
                error=str(error),
            )

    def _page_count(self) -> int:
        documents = list(discover_documents(self.material_folder))
        if not documents:
            raise RuntimeError(f"No PDF documents found in {self.material_folder}")
        total = 0
        for document in documents:
            with pymupdf.open(document.path) as pdf:
                total += len(pdf)
        return total

    def _is_fresh(self) -> bool:
        return self._database_is_complete(self.database_path)

    @staticmethod
    def _compatible_database(database_path: Path) -> bool:
        if not database_path.exists():
            return False
        try:
            with closing(sqlite3.connect(database_path)) as connection:
                row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = 'ingestion_schema_version'"
                ).fetchone()
                return bool(row and row[0] == INGESTION_SCHEMA_VERSION)
        except sqlite3.Error:
            return False

    def _prepare_staging(self, *, force: bool) -> None:
        if not force and self._compatible_database(self.staging_path):
            self._set(message="Resuming completed documents from the staging library...")
            return
        self._remove_staging_files()
        if not force and self._compatible_database(self.database_path):
            self._set(message="Reusing the active library as the incremental build baseline...")
            self._copy_database(self.database_path, self.staging_path)

    def _database_is_complete(self, database_path: Path) -> bool:
        if not database_path.exists():
            return False
        documents = list(discover_documents(self.material_folder))
        expected = {(str(item.path.resolve()), item.content_hash) for item in documents}
        try:
            with closing(sqlite3.connect(database_path)) as connection:
                schema_row = connection.execute(
                    "SELECT value FROM schema_metadata WHERE key = ?",
                    ("ingestion_schema_version",),
                ).fetchone()
                if schema_row is None or schema_row[0] != INGESTION_SCHEMA_VERSION:
                    return False
                stored = set(connection.execute("SELECT source_path, content_hash FROM documents"))
                if stored != expected:
                    return False
                pages = connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
                invalid_pages = _invalid_page_count(connection)
                invalid_processing = _invalid_document_processing_count(connection)
                chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
                embeddings = connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (EMBEDDING_MODEL,)
                ).fetchone()[0]
                blocks = connection.execute("SELECT COUNT(*) FROM page_blocks").fetchone()[0]
        except sqlite3.Error:
            return False
        return bool(
            pages
            and not invalid_pages
            and not invalid_processing
            and blocks
            and chunks
            and chunks == embeddings
        )

    def _activate_staging(self) -> None:
        """Activate a validated build through SQLite's online backup API.

        Status readers may briefly hold the stable database filename open.
        SQLite's backup API coordinates with those readers, whereas os.replace()
        fails on Windows as soon as any process has an open file handle.
        """
        self._validate_staging()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        if self.database_path.exists():
            self._copy_database(self.database_path, self.backup_path)
        self._copy_database(self.staging_path, self.database_path)
        if not self._database_is_complete(self.database_path):
            raise RuntimeError("The activated library did not pass validation")
        self._remove_database_files(self.staging_path)

    @staticmethod
    def _copy_database(source_path: Path, destination_path: Path) -> None:
        with closing(sqlite3.connect(source_path, timeout=30)) as source:
            with closing(sqlite3.connect(destination_path, timeout=30)) as destination:
                source.backup(destination)

    def _unload_ollama(self) -> None:
        try:
            models = self._ollama_models()
        except URLError:
            self._set(message="Ollama is not running; no model can be using its VRAM.")
            return
        if not models:
            self._set(message="No Ollama model is loaded.")
            return
        for model in models:
            self._check_cancelled()
            self._set(message=f"Unloading Ollama model: {model}")
            payload = json.dumps({"model": model, "keep_alive": 0, "stream": False}).encode("utf-8")
            request = Request(
                f"{OLLAMA_BASE_URL}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urlopen(request, timeout=10) as response:
                    response.read()
            except (HTTPError, URLError, TimeoutError) as error:
                raise RuntimeError(f"Could not unload Ollama model {model}: {error}") from error
        remaining = self._ollama_models()
        if remaining:
            raise RuntimeError("Ollama still reports loaded model(s): " + ", ".join(remaining))
        self._set(message="All Ollama models are unloaded; GPU memory is available for OCR.")

    @staticmethod
    def _ollama_models() -> tuple[str, ...]:
        request = Request(f"{OLLAMA_BASE_URL}/api/ps", method="GET")
        with urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return tuple(
            str(item.get("name") or item.get("model"))
            for item in payload.get("models", ())
            if item.get("name") or item.get("model")
        )

    def _run_command(self, command: list[str], *, total_pages: int = 0) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(PACKAGE_IMPORT_ROOT),
                "PYTHONIOENCODING": "utf-8",
                "HF_HUB_OFFLINE": "1",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            }
        )
        process = subprocess.Popen(
            command,
            cwd=self.project_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        with self._process_lock:
            self._process = process
        try:
            assert process.stdout is not None
            for raw_line in process.stdout:
                self._check_cancelled()
                line = raw_line.strip()
                if not line:
                    continue
                page_match = re.search(r"(?:OCR|Native|Cached) page (\d+)/(\d+):", line)
                chunk_match = re.search(r"Embedded (\d+)/(\d+) chunk", line)
                if page_match:
                    current = int(page_match.group(1))
                    total = int(page_match.group(2)) or total_pages
                    self._set(
                        stage="ocr",
                        message=line.removeprefix("[StudyRag] "),
                        current=current,
                        total=total,
                        percent=5 + int(75 * current / max(total, 1)),
                    )
                elif chunk_match:
                    current = int(chunk_match.group(1))
                    total = int(chunk_match.group(2))
                    self._set(
                        stage="indexing",
                        message=line.removeprefix("[StudyRag] "),
                        current=current,
                        total=total,
                        percent=84 + int(14 * current / max(total, 1)),
                    )
                else:
                    print(line, flush=True)
            return_code = process.wait()
        finally:
            with self._process_lock:
                if self._process is process:
                    self._process = None
        self._check_cancelled()
        if return_code:
            raise RuntimeError(f"Worker exited with code {return_code}")

    def _validate_staging(self) -> None:
        if not self.staging_path.exists():
            raise RuntimeError("The staging database was not created")
        with closing(sqlite3.connect(self.staging_path)) as connection:
            schema_row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = ?",
                ("ingestion_schema_version",),
            ).fetchone()
            pages = connection.execute("SELECT COUNT(*) FROM pages").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embeddings = connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE model_name = ?", (EMBEDDING_MODEL,)
            ).fetchone()[0]
            invalid_pages = _invalid_page_count(connection)
            invalid_processing = _invalid_document_processing_count(connection)
        if (
            schema_row is None
            or schema_row[0] != INGESTION_SCHEMA_VERSION
            or not pages
            or not chunks
            or chunks != embeddings
            or invalid_pages
            or invalid_processing
        ):
            raise RuntimeError(
                f"Staging validation failed: schema={schema_row}, pages={pages}, chunks={chunks}, "
                f"embeddings={embeddings}, invalid_pages={invalid_pages}, "
                f"invalid_processing={invalid_processing}"
            )
        if not self._database_is_complete(self.staging_path):
            raise RuntimeError("Staging validation failed: material changed during preparation")

    def _remove_staging_files(self) -> None:
        self._remove_database_files(self.staging_path)

    @staticmethod
    def _remove_database_files(database_path: Path) -> None:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(f"{database_path}{suffix}")
            if candidate.exists():
                candidate.unlink()
