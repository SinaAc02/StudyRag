from dataclasses import replace
from collections.abc import Callable
import unicodedata

from studyrag.domain.models import ExtractedPage, ExtractionResult, PageTextQuality
from studyrag.ingestion.assessment import assess_page
from studyrag.ingestion.ocr.base import OcrEngine
from studyrag.text.normalization import normalize_for_search


def apply_ocr_to_all_pages(
    extraction: ExtractionResult,
    engine: OcrEngine,
    languages: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
    page_offset: int = 0,
    total_pages: int | None = None,
    *,
    force_all: bool = True,
) -> tuple[ExtractionResult, int]:
    """Preserve native evidence; optionally route usable pages around OCR."""

    updated_pages = []
    ocr_count = 0
    validated = False
    page_count = total_pages or len(extraction.pages)
    for page in extraction.pages:
        reasons = ocr_reasons(page) if not force_all else ("forced",)
        if not reasons:
            if progress is not None:
                progress(
                    f"Native page {page_offset + page.page_number}/{page_count}: {page.document_path.name}"
                )
            updated_pages.append(page)
            continue
        if not validated:
            engine.validate(languages)
            validated = True
        if progress is not None:
            progress(
                f"OCR page {page_offset + page.page_number}/{page_count}: "
                f"{page.document_path.name}" + (f" ({', '.join(reasons)})" if not force_all else "")
            )
        result = engine.recognize_page(page.document_path, page.page_number, languages)
        updated_pages.append(
            replace(
                page,
                normalized_text=normalize_for_search(result.text),
                ocr_text=result.text,
                text_source="ocr",
                ocr_engine=result.engine,
                ocr_blocks=result.blocks,
            )
        )
        ocr_count += 1
    return replace(extraction, pages=tuple(updated_pages)), ocr_count


def ocr_reasons(page: ExtractedPage) -> tuple[str, ...]:
    """Choose full-page OCR conservatively; character count alone is insufficient."""
    reasons = list(page.visual_ocr_reasons)
    text = page.raw_text
    if assess_page(page).quality is not PageTextQuality.GOOD_NATIVE:
        reasons.append("insufficient_native_text")
    visible = [character for character in text if not character.isspace()]
    if visible:
        damaged = sum(
            character == "\ufffd" or unicodedata.category(character) in {"Co", "Cc", "Cs"}
            for character in visible
        )
        if damaged or sum(character.isalnum() for character in visible) / len(visible) < 0.5:
            reasons.append("damaged_native_text")
    if not page.native_blocks:
        reasons.append("missing_native_layout")
    return tuple(dict.fromkeys(reasons))


def apply_selective_ocr(
    extraction: ExtractionResult,
    engine: OcrEngine,
    languages: tuple[str, ...],
    progress: Callable[[str], None] | None = None,
    page_offset: int = 0,
    total_pages: int | None = None,
) -> tuple[ExtractionResult, int]:
    """Keep usable native blocks; lazily invoke OCR only for flagged pages."""
    return apply_ocr_to_all_pages(
        extraction,
        engine,
        languages,
        progress,
        page_offset,
        total_pages,
        force_all=False,
    )
