from pathlib import Path
import hashlib
import sqlite3

import numpy as np
import pymupdf
import pytest

from studyrag.domain.models import OcrBlock, OcrResult
from studyrag.ingestion.pipeline import ingest_folder
from studyrag.ingestion.ocr.paddle_structure import PaddleStructureEngine
from studyrag.preparation import PreparationCoordinator
from studyrag.retrieval.service import build_index


def write_pdf(path, text="A lesson about evidence and careful comparisons. " * 8):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 700), text)
        pdf.save(path)


class Ocr:
    name = PaddleStructureEngine.name
    cache_key = "test-configuration-v1"

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def validate(self, languages):
        return None

    def recognize_page(self, path, number, languages):
        self.calls.append(path.name)
        if path.name == self.fail_on:
            raise RuntimeError("Simulated interrupted OCR")
        text = f"Recognized text for {path.name}"
        return OcrResult(text, self.name, languages, (OcrBlock("text", text, (1, 1, 200, 80)),))


class Embedder:
    model_name = "intfloat/multilingual-e5-small"

    def __init__(self):
        self.texts = []

    def encode_passages(self, texts):
        self.texts.extend(texts)
        return np.ones((len(texts), 2), dtype=np.float32)


def test_unchanged_documents_skip_parser_ocr_and_embedding(tmp_path, monkeypatch):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "native.pdf")
    write_pdf(material / "scan.pdf", "")
    database = tmp_path / "index.sqlite"
    engine = Ocr()
    ingest_folder(material, database, ocr_engine=engine)
    build_index(database, Embedder())

    def forbidden_parse(*args):
        raise AssertionError("Unchanged documents must not be parsed again")

    monkeypatch.setattr("studyrag.ingestion.pipeline.NativePdfParser.parse", forbidden_parse)
    summary = ingest_folder(material, database, ocr_engine=engine)
    embedder = Embedder()
    index = build_index(database, embedder)
    assert summary.reused_document_count == 2
    assert summary.reused_page_count == 2
    assert summary.ocr_page_count == summary.native_page_count == 0
    assert engine.calls == ["scan.pdf"]
    assert embedder.texts == []
    assert index.reused_chunk_count == index.chunk_count


def test_add_change_and_remove_update_only_affected_documents(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf", "Original observations provide useful evidence. " * 8)
    database = tmp_path / "index.sqlite"
    ingest_folder(material, database, ocr_engine=Ocr())
    build_index(database, Embedder())

    write_pdf(material / "b.pdf", "Additional observations provide a second topic. " * 8)
    added = ingest_folder(material, database, ocr_engine=Ocr())
    encoder = Embedder()
    index = build_index(database, encoder)
    assert added.reused_document_count == 1
    assert len(encoder.texts) == 1
    assert index.reused_chunk_count == 1

    write_pdf(material / "b.pdf", "Changed material replaces the second topic. " * 8)
    changed = ingest_folder(material, database, ocr_engine=Ocr())
    encoder = Embedder()
    build_index(database, encoder)
    assert changed.reused_document_count == 1
    assert len(encoder.texts) == 1 and "Changed material" in encoder.texts[0]

    (material / "b.pdf").unlink()
    removed = ingest_folder(material, database, ocr_engine=Ocr())
    encoder = Embedder()
    build_index(database, encoder)
    assert removed.reused_document_count == 1
    assert encoder.texts == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1


def test_processing_configuration_change_invalidates_document_cache(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "scan.pdf", "")
    database = tmp_path / "index.sqlite"
    ingest_folder(material, database, ocr_engine=Ocr())
    engine = Ocr()
    engine.cache_key = "test-configuration-v2"
    summary = ingest_folder(material, database, ocr_engine=engine)
    assert summary.reused_document_count == 0
    assert engine.calls == ["scan.pdf"]


def test_interrupted_document_build_resumes_completed_documents(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf", "")
    write_pdf(material / "b.pdf", "")
    coordinator = PreparationCoordinator(tmp_path, material, tmp_path / "active.sqlite")
    with pytest.raises(RuntimeError, match="interrupted OCR"):
        ingest_folder(material, coordinator.staging_path, ocr_engine=Ocr(fail_on="b.pdf"))
    coordinator._prepare_staging(force=False)
    engine = Ocr()
    summary = ingest_folder(material, coordinator.staging_path, ocr_engine=engine)
    assert summary.reused_document_count == 1
    assert summary.ocr_page_count == 1
    assert engine.calls == ["b.pdf"]


def test_staging_clones_active_cache_and_force_build_discards_only_staging(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    active = tmp_path / "active.sqlite"
    ingest_folder(material, active, ocr_engine=Ocr())
    build_index(active, Embedder())
    active_hash = hashlib.sha256(active.read_bytes()).hexdigest()
    coordinator = PreparationCoordinator(tmp_path, material, active)
    coordinator._prepare_staging(force=False)
    summary = ingest_folder(material, coordinator.staging_path, ocr_engine=Ocr())
    assert summary.reused_document_count == 1
    coordinator._prepare_staging(force=True)
    assert not coordinator.staging_path.exists()
    assert hashlib.sha256(active.read_bytes()).hexdigest() == active_hash


def test_failed_incremental_build_keeps_active_database_intact(tmp_path, monkeypatch):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    active = tmp_path / "active.sqlite"
    ingest_folder(material, active, ocr_engine=Ocr())
    build_index(active, Embedder())
    original = active.read_bytes()
    write_pdf(material / "b.pdf")
    coordinator = PreparationCoordinator(tmp_path, material, active)
    monkeypatch.setattr(coordinator, "_unload_ollama", lambda: None)

    def worker(command, **kwargs):
        if "ingest" in command:
            ingest_folder(material, coordinator.staging_path, ocr_engine=Ocr())
        else:
            raise RuntimeError("Simulated embedding failure")

    monkeypatch.setattr(coordinator, "_run_command", worker)
    coordinator._run(force=False)
    assert coordinator.status()["stage"] == "error"
    assert active.read_bytes() == original
    engine = Ocr()
    resumed = ingest_folder(material, coordinator.staging_path, ocr_engine=engine)
    assert resumed.reused_document_count == 2
    assert engine.calls == []


def test_corrupt_cached_layout_is_repaired_instead_of_reused(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    database = tmp_path / "index.sqlite"
    ingest_folder(material, database, ocr_engine=Ocr())
    with sqlite3.connect(database) as connection:
        connection.execute("DELETE FROM page_blocks")
    summary = ingest_folder(material, database, ocr_engine=Ocr())
    assert summary.reused_document_count == 0
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM page_blocks").fetchone()[0] > 0


def test_material_changed_after_build_is_rejected_before_activation(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    pdf = material / "a.pdf"
    write_pdf(pdf)
    active = tmp_path / "active.sqlite"
    engine = PaddleStructureEngine()  # Native PDF: constructor only, no OCR runtime loaded.
    ingest_folder(material, active, ocr_engine=engine)
    build_index(active, Embedder())
    original = active.read_bytes()
    coordinator = PreparationCoordinator(tmp_path, material, active)
    coordinator._prepare_staging(force=False)
    write_pdf(pdf, "Changed evidence must invalidate the staging snapshot. " * 8)
    with pytest.raises(RuntimeError, match="material changed"):
        coordinator._activate_staging()
    assert active.read_bytes() == original


def test_processing_settings_change_invalidates_readiness(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    database = tmp_path / "index.sqlite"
    ingest_folder(material, database, ocr_engine=PaddleStructureEngine())
    build_index(database, Embedder())
    coordinator = PreparationCoordinator(tmp_path, material, database)
    assert coordinator._is_fresh() is True
    with sqlite3.connect(database) as connection:
        connection.execute("UPDATE document_processing SET processing_key = 'old-settings'")
    assert coordinator._is_fresh() is False


def test_previous_vl_backend_does_not_count_as_current_library(tmp_path):
    from studyrag.ingestion.ocr.paddle_vl import PaddleOcrVlEngine

    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    database = tmp_path / "index.sqlite"
    # Native text avoids loading either OCR runtime in this regression test.
    ingest_folder(material, database, ocr_engine=PaddleOcrVlEngine())
    build_index(database, Embedder())
    coordinator = PreparationCoordinator(tmp_path, material, database)
    assert coordinator._is_fresh() is False


def test_plus_s_processing_key_requires_rebuild_for_plus_l(tmp_path):
    material = tmp_path / "material"
    material.mkdir()
    write_pdf(material / "a.pdf")
    database = tmp_path / "index.sqlite"
    current = PaddleStructureEngine()

    class PreviousFormulaEngine:
        name = "pp-structure-v3-mobile-formula-s"
        cache_key = current.cache_key.replace("formula-l", "formula-s")

        def validate(self, languages):
            raise AssertionError("Native test PDF must not load OCR")

    ingest_folder(material, database, ocr_engine=PreviousFormulaEngine())
    build_index(database, Embedder())
    original = database.read_bytes()
    coordinator = PreparationCoordinator(tmp_path, material, database)
    assert coordinator._is_fresh() is False
    assert database.read_bytes() == original
