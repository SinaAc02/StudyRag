from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from hashlib import sha256
import json

import pymupdf

from studyrag.domain.models import PageTextQuality
from studyrag.ingestion.assessment import assess_page
from studyrag.ingestion.chunking import chunk_page
from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.ocr import OcrEngine
from studyrag.ingestion.ocr.routing import apply_selective_ocr
from studyrag.ingestion.parsers import NativePdfParser
from studyrag.ingestion.store import INGESTION_SCHEMA_VERSION, IngestionStore


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    document_count: int
    page_count: int
    chunk_count: int
    visual_page_count: int
    ocr_page_count: int
    quality_counts: dict[PageTextQuality, int]
    database_path: Path
    reused_document_count: int = 0
    reused_page_count: int = 0
    native_page_count: int = 0


def processing_key_for(engine: OcrEngine | None, languages: tuple[str, ...]) -> str:
    configuration = {
        "policy": INGESTION_SCHEMA_VERSION,
        "pymupdf": pymupdf.VersionBind,
        "ocr": getattr(engine, "cache_key", getattr(engine, "name", None)),
        "languages": languages,
    }
    return sha256(json.dumps(configuration, sort_keys=True).encode("utf-8")).hexdigest()


def ingest_folder(
    folder: Path,
    database_path: Path,
    *,
    ocr_engine: OcrEngine | None = None,
    ocr_languages: tuple[str, ...] = ("fas", "eng", "deu"),
    progress: Callable[[str], None] | None = None,
) -> IngestionSummary:
    documents = list(discover_documents(folder))
    parser = NativePdfParser()
    store = IngestionStore(database_path)
    store.initialize()
    store.remove_missing_documents({str(document.path.resolve()) for document in documents})
    processing_key = processing_key_for(ocr_engine, ocr_languages)

    quality_counts: Counter[PageTextQuality] = Counter()
    total_pages = 0
    total_chunks = 0
    visual_pages = 0
    ocr_pages = 0
    native_pages = 0
    reused_documents = 0
    reused_pages = 0
    if progress is not None:
        progress(f"Found {len(documents)} PDF document(s).")
    plans = []
    all_page_count = 0
    for document in documents:
        cached = store.cached_document(document, processing_key)
        extraction = None if cached is not None else parser.parse(document)
        all_page_count += cached.page_count if cached else len(extraction.pages)
        plans.append((document, extraction, cached))
    completed_pages = 0
    for document_index, (document, extraction, cached) in enumerate(plans, start=1):
        if progress is not None:
            progress(f"Document {document_index}/{len(documents)}: {document.path.name}")
        if cached is not None:
            reused_documents += 1
            reused_pages += cached.page_count
            completed_pages += cached.page_count
            total_pages += cached.page_count
            total_chunks += cached.chunk_count
            visual_pages += cached.visual_page_count
            quality_counts.update(cached.quality_counts)
            if progress is not None:
                progress(
                    f"Cached page {completed_pages}/{all_page_count}: "
                    f"reused {document.path.name} ({cached.page_count} page(s))"
                )
            continue
        assert extraction is not None
        document_ocr_pages = 0
        if ocr_engine is not None:
            extraction, document_ocr_pages = apply_selective_ocr(
                extraction,
                ocr_engine,
                ocr_languages,
                progress,
                completed_pages,
                all_page_count,
            )
            ocr_pages += document_ocr_pages
        assessments = tuple(assess_page(page) for page in extraction.pages)
        chunks = tuple(chunk for page in extraction.pages for chunk in chunk_page(document, page))
        store.replace_document(extraction, assessments, chunks, processing_key=processing_key)
        native_pages += len(extraction.pages) - document_ocr_pages
        if progress is not None:
            progress(
                f"Stored {len(extraction.pages)} page(s) and {len(chunks)} chunk(s) "
                f"from {document.path.name}."
            )
        total_pages += len(extraction.pages)
        total_chunks += len(chunks)
        visual_pages += sum(item.visually_important for item in assessments)
        quality_counts.update(item.quality for item in assessments)
        completed_pages += len(extraction.pages)

    summary = IngestionSummary(
        document_count=len(documents),
        page_count=total_pages,
        chunk_count=total_chunks,
        visual_page_count=visual_pages,
        ocr_page_count=ocr_pages,
        quality_counts=dict(quality_counts),
        database_path=store.database_path,
        reused_document_count=reused_documents,
        reused_page_count=reused_pages,
        native_page_count=native_pages,
    )
    if progress is not None:
        progress(
            f"Ingestion complete: {summary.document_count} document(s), "
            f"{summary.page_count} page(s), {summary.chunk_count} chunk(s)."
        )
    return summary
