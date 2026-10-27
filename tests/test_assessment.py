from pathlib import Path

from studyrag.domain.models import ExtractedPage, PageTextQuality
from studyrag.ingestion.assessment import assess_page


def _page(text: str, *, images: int = 0, drawings: int = 0) -> ExtractedPage:
    return ExtractedPage(Path("lesson.pdf"), 1, text, text, images, drawings)


def test_empty_page_requires_ocr() -> None:
    assert assess_page(_page("")).quality is PageTextQuality.OCR_NEEDED


def test_sparse_page_is_not_misreported_as_good() -> None:
    assert assess_page(_page("short text" * 3)).quality is PageTextQuality.SPARSE_NATIVE


def test_visual_importance_is_separate_from_text_quality() -> None:
    assessment = assess_page(_page("useful text " * 20, images=1))
    assert assessment.quality is PageTextQuality.GOOD_NATIVE
    assert assessment.visually_important is True
    assert "contains_images" in assessment.reasons
