from collections import Counter
from pathlib import Path
import sqlite3

import pymupdf
import pytest

from studyrag.domain.models import (
    ExtractedPage,
    ExtractionResult,
    OcrBlock,
    OcrResult,
    SourceDocument,
)
from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.ocr.routing import apply_selective_ocr, ocr_reasons
from studyrag.ingestion.ocr.paddle_structure import PaddleStructureEngine
from studyrag.ingestion.parsers import NativePdfParser
from studyrag.ingestion.pipeline import ingest_folder
from studyrag.preparation import PreparationCoordinator
from studyrag.retrieval.service import build_index


TEXT = "A native PDF lesson explains correlation, observations and careful comparisons. " * 4
LANGUAGES = ("fas", "eng", "deu")


class RecordingOcr:
    name = PaddleStructureEngine.name
    cache_key = PaddleStructureEngine().cache_key

    def __init__(self):
        self.pages = []
        self.validations = 0

    def validate(self, languages):
        self.validations += 1

    def recognize_page(self, pdf_path, page_number, languages):
        self.pages.append(page_number)
        return OcrResult(
            "Recognized scan text",
            self.name,
            languages,
            (OcrBlock("text", "Recognized scan text", (20, 20, 200, 80), 0),),
        )


def _pdf(path: Path, *, image=False, vector=False, logo=False, background=False, scan=False):
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(pymupdf.Rect(40, 40, 550, 250), TEXT)
        if image or logo:
            pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 40, 40), False)
            pixmap.clear_with(120)
            rect = pymupdf.Rect(40, 300, 300, 500) if image else pymupdf.Rect(5, 5, 15, 15)
            page.insert_image(rect, stream=pixmap.tobytes("png"))
        if vector:
            page.draw_rect(pymupdf.Rect(40, 300, 300, 500))
            page.draw_line((40, 300), (300, 500))
        if background:
            page.draw_rect(page.rect, color=None, fill=(0.9, 0.9, 0.9), overlay=False)
        if scan:
            pdf.new_page()
        pdf.save(path)
    return next(discover_documents(path.parent))


def test_real_native_pdf_keeps_layout_and_never_initializes_ocr(tmp_path):
    document = _pdf(tmp_path / "lesson.pdf")
    parsed = NativePdfParser().parse(document)
    assert parsed.pages[0].native_blocks
    assert parsed.pages[0].native_blocks[0].bbox[2] > 40
    assert ocr_reasons(parsed.pages[0]) == ()

    class UnavailableOcr(RecordingOcr):
        def validate(self, languages):
            raise AssertionError("Native pages must not require the OCR runtime")

    database = tmp_path / "native.sqlite"
    summary = ingest_folder(tmp_path, database, ocr_engine=UnavailableOcr())
    assert summary.ocr_page_count == 0
    assert summary.chunk_count > 0
    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT text_source, ocr_engine FROM pages").fetchall()
        assert rows == [("native", None)]
        assert connection.execute("SELECT COUNT(*) FROM page_blocks").fetchone()[0] > 0


@pytest.mark.parametrize(
    "visual,reason",
    [
        ("image", "substantial_image"),
        ("vector", "substantial_vector_graphics"),
    ],
)
def test_substantial_visual_content_still_uses_ocr(tmp_path, visual, reason):
    document = _pdf(tmp_path / "lesson.pdf", **{visual: True})
    parsed = NativePdfParser().parse(document)
    assert reason in ocr_reasons(parsed.pages[0])
    engine = RecordingOcr()
    updated, count = apply_selective_ocr(parsed, engine, LANGUAGES)
    assert count == 1
    assert engine.pages == [1]
    assert updated.pages[0].raw_text == parsed.pages[0].raw_text
    assert updated.pages[0].layout_blocks == updated.pages[0].ocr_blocks


@pytest.mark.parametrize("decoration", ["logo", "background"])
def test_small_logo_or_plain_background_does_not_force_ocr(tmp_path, decoration):
    document = _pdf(tmp_path / "lesson.pdf", **{decoration: True})
    assert ocr_reasons(NativePdfParser().parse(document).pages[0]) == ()


@pytest.mark.parametrize(
    "text,reason",
    [
        ("short native text", "insufficient_native_text"),
        (TEXT + "\ufffd", "damaged_native_text"),
        (TEXT + "\ue000", "damaged_native_text"),
    ],
)
def test_weak_text_is_not_accepted_only_because_blocks_exist(text, reason):
    page = ExtractedPage(
        Path("lesson.pdf"),
        1,
        text,
        text,
        native_blocks=(OcrBlock("text", text, (20, 20, 500, 250), 0),),
    )
    assert reason in ocr_reasons(page)


def test_persian_native_text_is_preserved_without_ocr():
    text = "این متن فارسی درباره یادگیری و بررسی منابع نوشته شده است. " * 5
    page = ExtractedPage(
        Path("lesson.pdf"),
        1,
        text,
        text,
        native_blocks=(OcrBlock("text", text, (20, 20, 500, 250), 0),),
    )
    engine = RecordingOcr()
    extraction = ExtractionResult(SourceDocument(page.document_path, "hash", 1), (page,))
    updated, count = apply_selective_ocr(extraction, engine, LANGUAGES)
    assert count == 0 and engine.validations == 0
    assert updated.pages[0].layout_blocks[0].text == text


def test_mixed_routing_advances_global_progress_and_preserves_page_numbers(tmp_path):
    document = _pdf(tmp_path / "lesson.pdf")
    native = NativePdfParser().parse(document).pages[0]
    scan = ExtractedPage(document.path, 2, "", "")
    engine = RecordingOcr()
    messages = []
    updated, count = apply_selective_ocr(
        ExtractionResult(document, (native, scan)),
        engine,
        LANGUAGES,
        messages.append,
        page_offset=3,
        total_pages=9,
    )
    assert count == 1 and engine.pages == [2]
    assert [page.page_number for page in updated.pages] == [1, 2]
    assert Counter(page.text_source for page in updated.pages) == {"native": 1, "ocr": 1}
    assert messages[0].startswith("Native page 4/9:")
    assert messages[1].startswith("OCR page 5/9:")


class FakeEmbedder:
    model_name = "intfloat/multilingual-e5-small"

    def encode_passages(self, texts):
        import numpy as np

        return np.ones((len(texts), 2), dtype=np.float32)


@pytest.mark.parametrize("mixed", [False, True])
def test_native_and_mixed_libraries_validate_activate_and_remain_fresh(tmp_path, mixed):
    material = tmp_path / "material"
    material.mkdir()
    _pdf(material / "lesson.pdf", scan=mixed)
    active = tmp_path / "studyrag.sqlite"
    coordinator = PreparationCoordinator(tmp_path, material, active)
    summary = ingest_folder(material, coordinator.staging_path, ocr_engine=RecordingOcr())
    assert summary.ocr_page_count == int(mixed)
    build_index(coordinator.staging_path, FakeEmbedder())
    coordinator._validate_staging()
    coordinator._activate_staging()
    assert coordinator._is_fresh() is True

    # Each mutation below represents incomplete or inconsistent page provenance.
    with sqlite3.connect(active) as connection:
        connection.execute("UPDATE pages SET ocr_engine = NULL WHERE text_source = 'ocr'")
        if not mixed:
            connection.execute("DELETE FROM page_blocks")
    assert coordinator._is_fresh() is False


@pytest.mark.parametrize(
    "change",
    [
        "UPDATE pages SET text_source = 'ocr', ocr_engine = NULL",
        "UPDATE pages SET text_source = 'native', ocr_engine = 'unknown'",
        "UPDATE schema_metadata SET value = '2' WHERE key = 'ingestion_schema_version'",
    ],
)
def test_staging_rejects_incomplete_provenance_and_previous_policy(tmp_path, change):
    material = tmp_path / "material"
    material.mkdir()
    _pdf(material / "lesson.pdf")
    coordinator = PreparationCoordinator(tmp_path, material, tmp_path / "studyrag.sqlite")
    ingest_folder(material, coordinator.staging_path, ocr_engine=RecordingOcr())
    build_index(coordinator.staging_path, FakeEmbedder())
    with sqlite3.connect(coordinator.staging_path) as connection:
        connection.execute(change)
    with pytest.raises(RuntimeError, match="Staging validation failed"):
        coordinator._validate_staging()


def test_worker_progress_accepts_native_and_ocr_pages(tmp_path, monkeypatch):
    coordinator = PreparationCoordinator(tmp_path, tmp_path / "material", tmp_path / "db.sqlite")

    class Worker:
        stdout = iter(
            [
                "Native page 1/3: lesson.pdf\n",
                "OCR page 2/3: lesson.pdf\n",
                "Native page 3/3: lesson.pdf\n",
            ]
        )

        def wait(self):
            return 0

    monkeypatch.setattr("studyrag.preparation.subprocess.Popen", lambda *a, **kw: Worker())
    coordinator._run_command(["fake-worker"], total_pages=3)
    assert coordinator.status()["current"] == 3
    assert coordinator.status()["total"] == 3
    assert coordinator.status()["percent"] == 80
