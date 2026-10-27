from typing import Protocol

from studyrag.domain.models import ExtractionResult, SourceDocument


class DocumentParser(Protocol):
    def parse(self, document: SourceDocument) -> ExtractionResult: ...
