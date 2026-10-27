from pathlib import Path

from studyrag.domain.models import ExtractedPage, ExtractionResult, OcrResult, SourceDocument
from studyrag.domain.models import OcrBlock
from studyrag.ingestion.ocr.routing import apply_ocr_to_all_pages


class FakeOcrEngine:
    name = "fake-ocr"

    def __init__(self) -> None:
        self.pages: list[int] = []

    def validate(self, languages: tuple[str, ...]) -> None:
        assert languages == ("fas", "eng", "deu")

    def recognize_page(self, pdf_path: Path, page_number: int, languages: tuple[str, ...]) -> OcrResult:
        self.pages.append(page_number)
        return OcrResult(
            f"كتاب يادگيري {page_number}",
            self.name,
            languages,
            (OcrBlock("text", f"example {page_number}", (1, 2, 3, 4), 1),),
        )


def test_every_page_is_sent_to_layout_aware_ocr() -> None:
    path = Path("lesson.pdf")
    document = SourceDocument(path, "hash", 100)
    extraction = ExtractionResult(
        document,
        (
            ExtractedPage(path, 1, "", ""),
            ExtractedPage(path, 2, "useful " * 30, "useful " * 30),
        ),
    )
    engine = FakeOcrEngine()

    messages: list[str] = []
    updated, count = apply_ocr_to_all_pages(extraction, engine, ("fas", "eng", "deu"), messages.append)

    assert count == 2
    assert engine.pages == [1, 2]
    assert updated.pages[0].raw_text == ""
    assert updated.pages[0].ocr_text == "كتاب يادگيري 1"
    assert updated.pages[0].normalized_text == "کتاب یادگیری 1"
    assert updated.pages[0].text_source == "ocr"
    assert updated.pages[0].ocr_engine == "fake-ocr"
    assert updated.pages[1].text_source == "ocr"
    assert updated.pages[1].raw_text.startswith("useful")
    assert updated.pages[1].ocr_blocks[0].bbox == (1, 2, 3, 4)
    assert messages == ["OCR page 1/2: lesson.pdf", "OCR page 2/2: lesson.pdf"]
