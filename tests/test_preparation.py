from pathlib import Path
from threading import Event
import json

import numpy as np
import pymupdf

from studyrag.domain.models import (
    ExtractedPage,
    ExtractionResult,
    OcrBlock,
    PageAssessment,
    PageTextQuality,
)
from studyrag.ingestion.chunking import chunk_page
from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.store import IngestionStore
from studyrag.ingestion.pipeline import processing_key_for
from studyrag.ingestion.ocr.paddle_structure import PaddleStructureEngine
from studyrag.preparation import PreparationCoordinator
from studyrag.retrieval.service import build_index


class FakeEmbedder:
    model_name = "intfloat/multilingual-e5-small"

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.ones((len(texts), 2), dtype=np.float32)


def _make_ready_library(material: Path, database: Path) -> None:
    material.mkdir(exist_ok=True)
    pdf_path = material / "lesson.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Weighted average example")
        pdf.save(pdf_path)
    document = next(discover_documents(material))
    page = ExtractedPage(
        pdf_path,
        1,
        "Weighted average example",
        "Weighted average example",
        ocr_text="Weighted average example",
        text_source="ocr",
        ocr_engine=PaddleStructureEngine.name,
        ocr_blocks=(OcrBlock("text", "Weighted average example", (1, 2, 3, 4), 1),),
    )
    extraction = ExtractionResult(document, (page,))
    assessment = PageAssessment(PageTextQuality.GOOD_NATIVE, 24, False, ())
    store = IngestionStore(database)
    store.initialize()
    store.replace_document(
        extraction,
        (assessment,),
        chunk_page(document, page),
        processing_key=processing_key_for(PaddleStructureEngine(), ("fas", "eng", "deu")),
    )
    build_index(database, FakeEmbedder())


def test_freshness_requires_matching_material_and_complete_paddle_index(tmp_path: Path) -> None:
    material = tmp_path / "material"
    database = tmp_path / "studyrag.sqlite"
    _make_ready_library(material, database)
    coordinator = PreparationCoordinator(tmp_path, material, database)

    assert coordinator._is_fresh() is True

    with (material / "lesson.pdf").open("ab") as stream:
        stream.write(b"changed")
    assert coordinator._is_fresh() is False


def test_freshness_rejects_an_old_ingestion_schema(tmp_path: Path) -> None:
    material = tmp_path / "material"
    database = tmp_path / "studyrag.sqlite"
    _make_ready_library(material, database)

    import sqlite3

    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE schema_metadata SET value = '1' WHERE key = 'ingestion_schema_version'")

    coordinator = PreparationCoordinator(tmp_path, material, database)

    assert coordinator._is_fresh() is False


def test_only_one_preparation_job_can_start(tmp_path: Path) -> None:
    coordinator = PreparationCoordinator(tmp_path, tmp_path / "material", tmp_path / "studyrag.sqlite")
    release = Event()
    coordinator._run = lambda *, force: release.wait(2)

    assert coordinator.start() is True
    assert coordinator.start() is False
    release.set()


def test_completed_staging_database_is_recovered_without_reprocessing(tmp_path: Path) -> None:
    material = tmp_path / "material"
    active = tmp_path / "studyrag.sqlite"
    staging = tmp_path / "studyrag.building.sqlite"
    _make_ready_library(material, staging)
    coordinator = PreparationCoordinator(tmp_path, material, active)
    coordinator._unload_ollama = lambda: (_ for _ in ()).throw(
        AssertionError("Recovery must not start model work")
    )

    coordinator._run(force=False)

    assert coordinator.status()["ready"] is True
    assert coordinator._is_fresh() is True
    assert not staging.exists()


def test_activation_works_while_active_database_has_a_reader(tmp_path: Path) -> None:
    material = tmp_path / "material"
    active = tmp_path / "studyrag.sqlite"
    staging = tmp_path / "studyrag.building.sqlite"
    _make_ready_library(material, active)
    _make_ready_library(material, staging)
    coordinator = PreparationCoordinator(tmp_path, material, active)

    import sqlite3

    reader = sqlite3.connect(active)
    reader.execute("SELECT COUNT(*) FROM pages").fetchone()
    try:
        coordinator._activate_staging()
    finally:
        reader.close()

    assert coordinator._is_fresh() is True
    assert not staging.exists()


class FakeHttpResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


def test_ollama_unload_uses_bounded_local_api_for_every_loaded_model(tmp_path: Path, monkeypatch) -> None:
    coordinator = PreparationCoordinator(tmp_path, tmp_path / "material", tmp_path / "studyrag.sqlite")
    requests: list[tuple[str, bytes | None, float]] = []
    model_snapshots = iter(
        [
            {"models": [{"name": "qwen:latest"}, {"name": "embed:latest"}]},
            {"models": []},
        ]
    )

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.data, timeout))
        if request.full_url.endswith("/api/ps"):
            return FakeHttpResponse(next(model_snapshots))
        return FakeHttpResponse({"done": True})

    monkeypatch.setattr("studyrag.preparation.urlopen", fake_urlopen)

    coordinator._unload_ollama()

    unload_payloads = [json.loads(data) for url, data, _ in requests if url.endswith("/api/generate")]
    assert unload_payloads == [
        {"model": "qwen:latest", "keep_alive": 0, "stream": False},
        {"model": "embed:latest", "keep_alive": 0, "stream": False},
    ]
    assert all(timeout <= 10 for _, _, timeout in requests)


def test_cancel_terminates_the_active_worker(tmp_path: Path) -> None:
    coordinator = PreparationCoordinator(tmp_path, tmp_path / "material", tmp_path / "studyrag.sqlite")

    class Worker:
        terminated = False
        killed = False

        def poll(self):
            return 0 if self.terminated or self.killed else None

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    worker = Worker()
    coordinator._process = worker

    coordinator.cancel()

    assert worker.terminated is True
    assert worker.killed is False
