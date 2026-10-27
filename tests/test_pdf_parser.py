from pathlib import Path

import pymupdf

from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.parsers import NativePdfParser


def _write_pdf(path: Path, pages: list[str]) -> None:
    with pymupdf.open() as pdf:
        for text in pages:
            page = pdf.new_page()
            page.insert_text((72, 72), text)
        pdf.save(path)


def test_extracts_text_with_one_based_page_numbers(tmp_path: Path) -> None:
    pdf_path = tmp_path / "lesson.pdf"
    _write_pdf(pdf_path, ["First topic", "Second topic"])
    document = next(discover_documents(tmp_path))

    result = NativePdfParser().parse(document)

    assert [page.page_number for page in result.pages] == [1, 2]
    assert "First topic" in result.pages[0].raw_text
    assert "Second topic" in result.pages[1].normalized_text
    assert result.character_count > 0


def test_empty_image_style_page_is_reported_without_fake_text(tmp_path: Path) -> None:
    pdf_path = tmp_path / "scan.pdf"
    _write_pdf(pdf_path, [""])
    document = next(discover_documents(tmp_path))

    result = NativePdfParser().parse(document)

    assert len(result.pages) == 1
    assert result.pages[0].raw_text == ""
    assert result.pages[0].normalized_text == ""
