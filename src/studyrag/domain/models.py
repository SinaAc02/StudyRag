from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SourceDocument:
    path: Path
    content_hash: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    document_path: Path
    page_number: int
    raw_text: str
    normalized_text: str
    image_count: int = 0
    drawing_count: int = 0
    ocr_text: str | None = None
    text_source: str = "native"
    ocr_engine: str | None = None
    ocr_blocks: tuple["OcrBlock", ...] = ()
    native_blocks: tuple["OcrBlock", ...] = ()
    visual_ocr_reasons: tuple[str, ...] = ()

    @property
    def layout_blocks(self) -> tuple["OcrBlock", ...]:
        if self.text_source == "ocr":
            return self.ocr_blocks
        return self.native_blocks or self.ocr_blocks


@dataclass(frozen=True, slots=True)
class OcrBlock:
    label: str
    text: str
    bbox: tuple[int, int, int, int]
    order: int | None = None


@dataclass(frozen=True, slots=True)
class OcrResult:
    text: str
    engine: str
    languages: tuple[str, ...]
    blocks: tuple[OcrBlock, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    document: SourceDocument
    pages: tuple[ExtractedPage, ...]

    @property
    def character_count(self) -> int:
        return sum(len(page.raw_text) for page in self.pages)


class PageTextQuality(StrEnum):
    GOOD_NATIVE = "good_native"
    SPARSE_NATIVE = "sparse_native"
    OCR_NEEDED = "ocr_needed"


@dataclass(frozen=True, slots=True)
class PageAssessment:
    quality: PageTextQuality
    text_characters: int
    visually_important: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    document_hash: str
    document_path: Path
    page_number: int
    chunk_index: int
    text: str
    region_index: int = 0
