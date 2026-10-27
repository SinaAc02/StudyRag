from studyrag.domain.models import ExtractedPage, PageAssessment, PageTextQuality


def assess_page(
    page: ExtractedPage,
    *,
    ocr_threshold: int = 20,
    sparse_threshold: int = 120,
    drawing_threshold: int = 8,
) -> PageAssessment:
    """Classify pages by native text coverage and visual-content flags."""

    character_count = len(page.normalized_text)
    reasons: list[str] = list(page.visual_ocr_reasons)
    if character_count < ocr_threshold:
        quality = PageTextQuality.OCR_NEEDED
        reasons.append("too_little_native_text")
    elif character_count < sparse_threshold:
        quality = PageTextQuality.SPARSE_NATIVE
        reasons.append("sparse_native_text")
    else:
        quality = PageTextQuality.GOOD_NATIVE

    visually_important = page.image_count > 0 or page.drawing_count >= drawing_threshold
    if page.image_count:
        reasons.append("contains_images")
    if page.drawing_count >= drawing_threshold:
        reasons.append("many_vector_drawings")

    return PageAssessment(
        quality=quality,
        text_characters=character_count,
        visually_important=visually_important,
        reasons=tuple(reasons),
    )
