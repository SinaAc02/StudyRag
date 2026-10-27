from pathlib import Path

from studyrag.domain.models import ExtractedPage, OcrBlock, SourceDocument
from studyrag.ingestion.chunking import chunk_page


def test_chunks_stay_on_their_source_page_and_have_stable_ids() -> None:
    document = SourceDocument(Path("lesson.pdf"), "abc123", 100)
    page = ExtractedPage(Path("lesson.pdf"), 3, "", "a" * 100 + "\n" + "b" * 100)

    first = chunk_page(document, page, max_characters=120)
    second = chunk_page(document, page, max_characters=120)

    assert len(first) == 2
    assert first == second
    assert {chunk.page_number for chunk in first} == {3}
    assert all(len(chunk.text) <= 120 for chunk in first)


def test_empty_page_produces_no_fake_chunk() -> None:
    document = SourceDocument(Path("scan.pdf"), "scan123", 100)
    page = ExtractedPage(Path("scan.pdf"), 1, "", "")
    assert chunk_page(document, page) == ()


def test_nearby_layout_blocks_form_one_logical_region() -> None:
    document = SourceDocument(Path("slides.pdf"), "hash", 100)
    page = ExtractedPage(
        Path("slides.pdf"),
        5,
        "native page text",
        "native page text",
        ocr_blocks=(
            OcrBlock("text", "weighted average definition", (10, 10, 100, 50), 1),
            OcrBlock("formula", "Sp2 = 1376.41", (10, 60, 100, 90), 2),
        ),
    )

    chunks = chunk_page(document, page)

    assert [chunk.text for chunk in chunks] == ["weighted average definition\nSp2 = 1376.41"]
    assert chunks[0].region_index == 0


def test_large_whitespace_separates_visual_slide_regions() -> None:
    document = SourceDocument(Path("handout.pdf"), "hash", 100)
    page = ExtractedPage(
        Path("handout.pdf"),
        5,
        "native page text",
        "native page text",
        ocr_blocks=(
            OcrBlock("text", "top left definition", (10, 10, 300, 100), 1),
            OcrBlock("formula", "top left formula", (10, 120, 300, 200), 2),
            OcrBlock("text", "top right explanation", (700, 10, 1000, 100), 3),
            OcrBlock("text", "bottom left exercise", (10, 700, 300, 800), 4),
            OcrBlock("text", "bottom right values", (700, 700, 1000, 800), 5),
            OcrBlock("formula", "bottom right result", (700, 820, 1000, 900), 6),
        ),
    )

    chunks = chunk_page(document, page)

    assert [chunk.text for chunk in chunks] == [
        "top left definition\ntop left formula",
        "top right explanation",
        "bottom left exercise",
        "bottom right values\nbottom right result",
    ]
    assert [chunk.region_index for chunk in chunks] == [0, 1, 2, 3]
