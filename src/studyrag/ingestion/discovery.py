from collections.abc import Iterator
from hashlib import sha256
from pathlib import Path

from studyrag.domain.models import SourceDocument


SUPPORTED_EXTENSIONS = frozenset({".pdf"})


def hash_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def discover_documents(folder: Path) -> Iterator[SourceDocument]:
    folder = folder.expanduser().resolve()
    if not folder.exists():
        raise FileNotFoundError(f"Learning-material folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected a folder, received: {folder}")

    for path in sorted(folder.rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_file() and path.suffix.casefold() in SUPPORTED_EXTENSIONS:
            yield SourceDocument(
                path=path,
                content_hash=hash_file(path),
                size_bytes=path.stat().st_size,
            )
