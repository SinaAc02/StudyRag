from pathlib import Path
from typing import Protocol

from studyrag.domain.models import OcrResult


class OcrUnavailableError(RuntimeError):
    """Raised when an explicitly requested OCR engine cannot run."""


class OcrEngine(Protocol):
    @property
    def name(self) -> str: ...

    def validate(self, languages: tuple[str, ...]) -> None: ...

    def recognize_page(
        self,
        pdf_path: Path,
        page_number: int,
        languages: tuple[str, ...],
    ) -> OcrResult: ...
