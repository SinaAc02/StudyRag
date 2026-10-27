import pymupdf

from studyrag.domain.models import ExtractedPage, ExtractionResult, OcrBlock, SourceDocument
from studyrag.text.normalization import normalize_for_search


class NativePdfParser:
    """Fast path for PDFs that already contain a usable text layer."""

    def parse(self, document: SourceDocument) -> ExtractionResult:
        if document.path.suffix.casefold() != ".pdf":
            raise ValueError(f"NativePdfParser cannot parse: {document.path}")

        pages: list[ExtractedPage] = []
        try:
            with pymupdf.open(document.path) as pdf:
                for index, page in enumerate(pdf):
                    raw_text = page.get_text("text")
                    layout = page.get_text(
                        "dict",
                        flags=pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES,
                        sort=True,
                    )
                    blocks = tuple(
                        OcrBlock(
                            "text",
                            "\n".join(
                                "".join(span["text"] for span in line["spans"]) for line in block["lines"]
                            ).strip(),
                            tuple(round(value) for value in block["bbox"]),
                            order,
                        )
                        for order, block in enumerate(layout["blocks"])
                        if block["type"] == 0
                    )
                    images = page.get_image_info()
                    drawings = page.get_drawings()
                    pages.append(
                        ExtractedPage(
                            document_path=document.path,
                            page_number=index + 1,
                            raw_text=raw_text,
                            normalized_text=normalize_for_search(raw_text),
                            image_count=len(images),
                            drawing_count=len(drawings),
                            native_blocks=tuple(block for block in blocks if block.text),
                            visual_ocr_reasons=_visual_ocr_reasons(page, layout, images, drawings),
                        )
                    )
        except (pymupdf.FileDataError, RuntimeError) as error:
            raise ValueError(f"Could not read PDF {document.path}: {error}") from error

        return ExtractionResult(document=document, pages=tuple(pages))


def _visual_ocr_reasons(page, layout, images, drawings) -> tuple[str, ...]:
    """Conservatively retain visual parsing for substantive non-text content."""
    reasons = []
    page_area = max(page.rect.get_area(), 1)

    def substantial(rect) -> bool:
        clipped = pymupdf.Rect(rect) & page.rect
        return min(clipped.width, clipped.height) >= 18 and clipped.get_area() / page_area >= 0.02

    if any(substantial(image["bbox"]) for image in images):
        reasons.append("substantial_image")
    # Ignore simple page-sized background fills/borders, not diagram components.
    content_drawings = [
        drawing
        for drawing in drawings
        if not (drawing["rect"].get_area() / page_area >= 0.85 and len(drawing["items"]) <= 1)
    ]
    if content_drawings and any(
        substantial(rect) for rect in page.cluster_drawings(drawings=content_drawings)
    ):
        reasons.append("substantial_vector_graphics")
    spans = [
        span
        for block in layout["blocks"]
        if block["type"] == 0
        for line in block["lines"]
        for span in line["spans"]
    ]
    if any(
        any(token in span["font"].casefold() for token in ("math", "symbol", "cmmi", "cmsy", "cmex"))
        or any(character in span["text"] for character in "∑∏∫√∂∇∈∉⊂⊆∞≈≠≤≥")
        for span in spans
    ):
        reasons.append("mathematical_content")
    return tuple(reasons)
