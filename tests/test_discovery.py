from pathlib import Path

import pytest

from studyrag.ingestion.discovery import discover_documents, hash_file


def test_discovers_only_pdfs_recursively(tmp_path: Path) -> None:
    nested = tmp_path / "course"
    nested.mkdir()
    pdf = nested / "lesson.PDF"
    pdf.write_bytes(b"same content")
    (nested / "notes.txt").write_text("ignored", encoding="utf-8")

    documents = list(discover_documents(tmp_path))

    assert [item.path for item in documents] == [pdf]
    assert documents[0].content_hash == hash_file(pdf)
    assert documents[0].size_bytes == len(b"same content")


def test_hash_changes_when_content_changes(tmp_path: Path) -> None:
    document = tmp_path / "lesson.pdf"
    document.write_bytes(b"version one")
    first_hash = hash_file(document)
    document.write_bytes(b"version two")
    assert hash_file(document) != first_hash


def test_missing_folder_has_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        list(discover_documents(tmp_path / "missing"))
