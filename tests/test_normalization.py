from studyrag.text.normalization import normalize_for_search


def test_normalizes_arabic_variants_used_in_persian() -> None:
    assert normalize_for_search("كتاب يادي") == "کتاب یادی"


def test_normalizes_alef_variants_for_search() -> None:
    assert normalize_for_search("آمور أمور امور إمر ٱمر") == "امور امور امور امر امر"


def test_preserves_line_boundaries_and_collapses_horizontal_spaces() -> None:
    assert normalize_for_search("  first   line\n  خط  دوم  ") == "first line\nخط دوم"


def test_removes_directional_marks_from_search_copy() -> None:
    assert normalize_for_search("هوش\u200f مصنوعی") == "هوش مصنوعی"
