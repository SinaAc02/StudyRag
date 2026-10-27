import re
import unicodedata


_CHARACTER_TRANSLATION = str.maketrans(
    {
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "آ": "ا",
        "ي": "ی",
        "ى": "ی",
        "ك": "ک",
        "ۀ": "هٔ",
        "ة": "ه",
    }
)
_ZERO_WIDTH_RE = re.compile("[\u200b\u200e\u200f\ufeff]")
_HORIZONTAL_SPACE_RE = re.compile(r"[^\S\r\n]+")


def normalize_for_search(text: str) -> str:
    """Normalize multilingual text for indexing while retaining line boundaries.

    Raw extracted text is stored separately for source citations.
    """

    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.translate(_CHARACTER_TRANSLATION)
    normalized = _ZERO_WIDTH_RE.sub("", normalized)
    normalized = _HORIZONTAL_SPACE_RE.sub(" ", normalized)
    return "\n".join(line.strip() for line in normalized.splitlines()).strip()
