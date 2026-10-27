from hashlib import sha256
from dataclasses import dataclass

from studyrag.domain.models import ExtractedPage, SourceDocument, TextChunk
from studyrag.text.normalization import normalize_for_search


NON_CONTENT_LABELS = {"header", "footer", "number", "page_number"}
REGION_GAP_RATIO = 0.07


@dataclass(frozen=True, slots=True)
class _LayoutUnit:
    text: str
    bbox: tuple[int, int, int, int]
    order: int


def _split_long_line(line: str, max_characters: int) -> list[str]:
    return [line[start : start + max_characters] for start in range(0, len(line), max_characters)]


def _best_whitespace_split(
    units: list[_LayoutUnit],
    page_width: int,
    page_height: int,
) -> tuple[str, int] | None:
    candidates: list[tuple[float, str, int]] = []
    for axis, page_span in (("x", page_width), ("y", page_height)):
        start_index, end_index = (0, 2) if axis == "x" else (1, 3)
        ordered = sorted(units, key=lambda unit: unit.bbox[start_index])
        furthest_end = ordered[0].bbox[end_index]
        for index, unit in enumerate(ordered[1:], start=1):
            gap = unit.bbox[start_index] - furthest_end
            if gap > 0:
                candidates.append((gap / max(page_span, 1), axis, index))
            furthest_end = max(furthest_end, unit.bbox[end_index])
    if not candidates:
        return None
    ratio, axis, index = max(candidates)
    return (axis, index) if ratio >= REGION_GAP_RATIO else None


def _layout_regions(units: list[_LayoutUnit]) -> list[list[_LayoutUnit]]:
    """Split a page on large empty bands while keeping nearby blocks together."""

    if not units:
        return []
    page_width = max(unit.bbox[2] for unit in units) - min(unit.bbox[0] for unit in units)
    page_height = max(unit.bbox[3] for unit in units) - min(unit.bbox[1] for unit in units)

    def split(group: list[_LayoutUnit]) -> list[list[_LayoutUnit]]:
        if len(group) < 3:
            return [group]
        boundary = _best_whitespace_split(group, page_width, page_height)
        if boundary is None:
            return [group]
        axis, index = boundary
        coordinate = 0 if axis == "x" else 1
        ordered = sorted(group, key=lambda unit: unit.bbox[coordinate])
        return split(ordered[:index]) + split(ordered[index:])

    regions = split(units)
    return sorted(
        regions,
        key=lambda region: (
            min(unit.bbox[1] for unit in region),
            min(unit.bbox[0] for unit in region),
        ),
    )


def _pack_units(units: list[str], max_characters: int) -> list[str]:
    texts: list[str] = []
    current: list[str] = []
    current_length = 0
    for unit in units:
        pieces = _split_long_line(unit, max_characters)
        for piece in pieces:
            separator_length = 1 if current else 0
            if current and current_length + separator_length + len(piece) > max_characters:
                texts.append("\n".join(current))
                current = []
                current_length = 0
                separator_length = 0
            current.append(piece)
            current_length += separator_length + len(piece)
    if current:
        texts.append("\n".join(current))
    return texts


def chunk_page(
    document: SourceDocument,
    page: ExtractedPage,
    *,
    max_characters: int = 1200,
) -> tuple[TextChunk, ...]:
    """Create deterministic chunks contained within one source page."""

    if max_characters < 100:
        raise ValueError("max_characters must be at least 100")
    layout_units = [
        _LayoutUnit(
            normalize_for_search(block.text).strip(),
            block.bbox,
            block.order if block.order is not None else index,
        )
        for index, block in enumerate(page.layout_blocks)
        if block.text.strip() and block.label not in NON_CONTENT_LABELS
    ]
    if not layout_units and not page.normalized_text.strip():
        return ()

    if layout_units:
        region_texts = [
            _pack_units(
                [unit.text for unit in sorted(region, key=lambda unit: unit.order)],
                max_characters,
            )
            for region in _layout_regions(layout_units)
        ]
    else:
        clean_lines = [line.strip() for line in page.normalized_text.splitlines() if line.strip()]
        region_texts = [_pack_units(clean_lines, max_characters)]

    chunks = []
    chunk_index = 0
    for region_index, texts in enumerate(region_texts):
        for text in texts:
            identity = f"{document.content_hash}:{page.page_number}:{region_index}:{chunk_index}:{text}"
            chunks.append(
                TextChunk(
                    chunk_id=sha256(identity.encode("utf-8")).hexdigest(),
                    document_hash=document.content_hash,
                    document_path=document.path,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=text,
                    region_index=region_index,
                )
            )
            chunk_index += 1
    return tuple(chunks)
