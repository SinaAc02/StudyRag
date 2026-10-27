from pathlib import Path

import numpy as np

from studyrag.domain.models import (
    ExtractedPage,
    ExtractionResult,
    PageAssessment,
    PageTextQuality,
    SourceDocument,
    OcrBlock,
)
from studyrag.ingestion.chunking import chunk_page
from studyrag.ingestion.store import IngestionStore
from studyrag.retrieval.service import build_index, search_index


class FakeEmbedder:
    model_name = "fake-multilingual"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        lowered = text.casefold()
        if "correlation" in lowered or "همبستگی" in lowered:
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


class AlefSensitiveEmbedder:
    model_name = "fake-alef-sensitive"

    @staticmethod
    def _vector(text: str) -> np.ndarray:
        if "امور" in text.casefold():
            return np.array([1.0, 0.0], dtype=np.float32)
        return np.array([0.0, 1.0], dtype=np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(text) for text in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._vector(text)


def _store_document(database: Path, path: Path, content_hash: str, text: str) -> None:
    document = SourceDocument(path, content_hash, len(text))
    page = ExtractedPage(path, 1, text, text)
    extraction = ExtractionResult(document, (page,))
    assessment = PageAssessment(PageTextQuality.GOOD_NATIVE, len(text), False, ())
    chunks = chunk_page(document, page)
    store = IngestionStore(database)
    store.initialize()
    store.replace_document(extraction, (assessment,), chunks)


def test_index_and_search_return_ranked_page_citations(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    _store_document(database, tmp_path / "correlation.pdf", "hash-a", "Correlation measures association.")
    _store_document(database, tmp_path / "anova.pdf", "hash-b", "ANOVA compares group means.")
    embedder = FakeEmbedder()

    summary = build_index(database, embedder)
    results = search_index(database, embedder, "همبستگی چیست؟", top_k=2)

    assert summary.chunk_count == 2
    assert summary.dimensions == 2
    assert results[0].source_path.name == "correlation.pdf"
    assert results[0].page_number == 1
    assert results[0].score == 1.0


def test_reindex_replaces_vectors_instead_of_duplicating(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    _store_document(database, tmp_path / "lesson.pdf", "hash-a", "Correlation topic")
    embedder = FakeEmbedder()
    build_index(database, embedder)
    build_index(database, embedder)

    results = search_index(database, embedder, "correlation", top_k=5)
    assert len(results) == 1


def test_index_reports_embedding_and_save_progress(tmp_path: Path) -> None:
    database = tmp_path / "index.sqlite"
    _store_document(database, tmp_path / "lesson.pdf", "hash-a", "Correlation topic")
    messages: list[str] = []

    build_index(database, FakeEmbedder(), messages.append)

    assert messages[0] == "Embedded 1/1 chunk(s) with fake-multilingual."
    assert messages[1] == "Saving 1 embedding vector(s) to SQLite..."
    assert messages[-1].startswith("Index complete:")


def test_keyword_search_breaks_a_semantic_tie(tmp_path: Path) -> None:
    database = tmp_path / "hybrid.sqlite"
    _store_document(database, tmp_path / "generic.pdf", "hash-a", "A general statistics lesson.")
    _store_document(
        database,
        tmp_path / "weighted.pdf",
        "hash-b",
        "Weighted average worked example with sample sizes.",
    )
    embedder = FakeEmbedder()
    build_index(database, embedder)

    results = search_index(database, embedder, "weighted average example", top_k=2)

    assert results[0].source_path.name == "weighted.pdf"


def test_search_normalizes_query_alef_variants_for_embeddings_and_keywords(
    tmp_path: Path,
) -> None:
    database = tmp_path / "persian.sqlite"
    _store_document(database, tmp_path / "other.pdf", "hash-a", "ماهی قزل آلا")
    _store_document(database, tmp_path / "amur.pdf", "hash-b", "ماهی امور")
    embedder = AlefSensitiveEmbedder()
    build_index(database, embedder)

    results = search_index(database, embedder, "ماهی آمور", top_k=2)

    assert results[0].source_path.name == "amur.pdf"


def test_search_expands_all_chunks_from_the_matching_region(tmp_path: Path) -> None:
    database = tmp_path / "expanded.sqlite"
    path = tmp_path / "lesson.pdf"
    document = SourceDocument(path, "hash", 400)
    page = ExtractedPage(
        path,
        1,
        "native",
        "native",
        ocr_blocks=(
            OcrBlock("text", "Weighted average definition " + "a" * 80, (10, 10, 200, 80), 1),
            OcrBlock("formula", "calculation result 1376.41 " + "b" * 80, (10, 90, 200, 150), 2),
        ),
    )
    extraction = ExtractionResult(document, (page,))
    assessment = PageAssessment(PageTextQuality.GOOD_NATIVE, 6, False, ())
    chunks = chunk_page(document, page, max_characters=100)
    store = IngestionStore(database)
    store.initialize()
    store.replace_document(extraction, (assessment,), chunks)
    build_index(database, FakeEmbedder())

    results = search_index(database, FakeEmbedder(), "1376.41", top_k=1)

    assert len(chunks) > 1
    assert "Weighted average definition" in results[0].text
    assert "calculation result 1376.41" in results[0].text
