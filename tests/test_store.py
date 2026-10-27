from pathlib import Path
import sqlite3

from studyrag.domain.models import (
    ExtractedPage,
    ExtractionResult,
    PageAssessment,
    PageTextQuality,
    SourceDocument,
)
from studyrag.ingestion.chunking import chunk_page
from studyrag.ingestion.store import IngestionStore


def test_reingestion_replaces_rows_instead_of_duplicating(tmp_path: Path) -> None:
    pdf_path = tmp_path / "lesson.pdf"
    pdf_path.write_bytes(b"document")
    document = SourceDocument(pdf_path, "hash-one", 8)
    page = ExtractedPage(pdf_path, 1, "text", "text")
    extraction = ExtractionResult(document, (page,))
    assessment = PageAssessment(PageTextQuality.SPARSE_NATIVE, 4, False, ("sparse_native_text",))
    chunks = chunk_page(document, page)
    store = IngestionStore(tmp_path / "index.sqlite")
    store.initialize()

    store.replace_document(extraction, (assessment,), chunks)
    store.replace_document(extraction, (assessment,), chunks)

    assert store.counts() == (1, 1, 1)
    with sqlite3.connect(tmp_path / "index.sqlite") as connection:
        assert connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0] == 1
