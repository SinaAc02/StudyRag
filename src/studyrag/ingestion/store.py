import sqlite3
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from dataclasses import dataclass

from studyrag.domain.models import (
    ExtractionResult,
    PageAssessment,
    PageTextQuality,
    SourceDocument,
    TextChunk,
)


INGESTION_SCHEMA_VERSION = "4"


@dataclass(frozen=True, slots=True)
class CachedDocument:
    page_count: int
    chunk_count: int
    visual_page_count: int
    quality_counts: dict[PageTextQuality, int]


class IngestionStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.executescript(
                """
                PRAGMA foreign_keys = ON;
                CREATE TABLE IF NOT EXISTS documents (
                    content_hash TEXT PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    size_bytes INTEGER NOT NULL,
                    page_count INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    document_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    raw_text TEXT NOT NULL,
                    normalized_text TEXT NOT NULL,
                    text_quality TEXT NOT NULL,
                    visually_important INTEGER NOT NULL,
                    assessment_reasons TEXT NOT NULL,
                    image_count INTEGER NOT NULL,
                    drawing_count INTEGER NOT NULL,
                    ocr_text TEXT,
                    text_source TEXT NOT NULL DEFAULT 'native',
                    ocr_engine TEXT,
                    PRIMARY KEY (document_hash, page_number),
                    FOREIGN KEY (document_hash) REFERENCES documents(content_hash)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    region_index INTEGER NOT NULL DEFAULT 0,
                    text TEXT NOT NULL,
                    FOREIGN KEY (document_hash, page_number)
                        REFERENCES pages(document_hash, page_number) ON DELETE CASCADE,
                    UNIQUE (document_hash, page_number, chunk_index)
                );
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_processing (
                    document_hash TEXT PRIMARY KEY,
                    processing_key TEXT NOT NULL,
                    FOREIGN KEY (document_hash) REFERENCES documents(content_hash)
                        ON DELETE CASCADE
                );
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                );
                CREATE TABLE IF NOT EXISTS page_blocks (
                    document_hash TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    block_index INTEGER NOT NULL,
                    label TEXT NOT NULL,
                    text TEXT NOT NULL,
                    x1 INTEGER NOT NULL,
                    y1 INTEGER NOT NULL,
                    x2 INTEGER NOT NULL,
                    y2 INTEGER NOT NULL,
                    reading_order INTEGER,
                    PRIMARY KEY (document_hash, page_number, block_index),
                    FOREIGN KEY (document_hash, page_number)
                        REFERENCES pages(document_hash, page_number) ON DELETE CASCADE
                );
                """
            )
            self._ensure_page_columns(connection)
            self._ensure_chunk_columns(connection)
            connection.execute(
                "INSERT OR REPLACE INTO schema_metadata(key, value) VALUES (?, ?)",
                ("ingestion_schema_version", INGESTION_SCHEMA_VERSION),
            )

    def replace_document(
        self,
        extraction: ExtractionResult,
        assessments: Iterable[PageAssessment],
        chunks: Iterable[TextChunk],
        *,
        processing_key: str | None = None,
    ) -> None:
        assessment_list = list(assessments)
        chunk_list = list(chunks)
        if len(assessment_list) != len(extraction.pages):
            raise ValueError("Every extracted page requires exactly one assessment")

        document = extraction.document
        source_path = str(document.path.resolve())
        with self._connection() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            previous = connection.execute(
                "SELECT content_hash FROM documents WHERE source_path = ?",
                (source_path,),
            ).fetchone()
            if previous and previous[0] != document.content_hash:
                connection.execute("DELETE FROM documents WHERE content_hash = ?", (previous[0],))
            connection.execute("DELETE FROM documents WHERE content_hash = ?", (document.content_hash,))
            connection.execute(
                "INSERT INTO documents(content_hash, source_path, size_bytes, page_count) "
                "VALUES (?, ?, ?, ?)",
                (document.content_hash, source_path, document.size_bytes, len(extraction.pages)),
            )
            for page, assessment in zip(extraction.pages, assessment_list, strict=True):
                connection.execute(
                    "INSERT INTO pages("
                    "document_hash, page_number, raw_text, normalized_text, text_quality, "
                    "visually_important, assessment_reasons, image_count, drawing_count, "
                    "ocr_text, text_source, ocr_engine"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        document.content_hash,
                        page.page_number,
                        page.raw_text,
                        page.normalized_text,
                        assessment.quality.value,
                        int(assessment.visually_important),
                        ",".join(assessment.reasons),
                        page.image_count,
                        page.drawing_count,
                        page.ocr_text,
                        page.text_source,
                        page.ocr_engine,
                    ),
                )
            connection.executemany(
                "INSERT INTO page_blocks(document_hash, page_number, block_index, label, "
                "text, x1, y1, x2, y2, reading_order) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        document.content_hash,
                        page.page_number,
                        index,
                        block.label,
                        block.text,
                        *block.bbox,
                        block.order,
                    )
                    for page in extraction.pages
                    for index, block in enumerate(page.layout_blocks)
                ],
            )
            connection.executemany(
                "INSERT INTO chunks("
                "chunk_id, document_hash, page_number, chunk_index, region_index, text"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.chunk_id,
                        chunk.document_hash,
                        chunk.page_number,
                        chunk.chunk_index,
                        chunk.region_index,
                        chunk.text,
                    )
                    for chunk in chunk_list
                ],
            )
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("INSERT INTO chunks_fts(chunk_id, text) SELECT chunk_id, text FROM chunks")
            if processing_key is not None:
                connection.execute(
                    "INSERT INTO document_processing(document_hash, processing_key) VALUES (?, ?)",
                    (document.content_hash, processing_key),
                )

    def cached_document(
        self,
        document: SourceDocument,
        processing_key: str,
    ) -> CachedDocument | None:
        with self._connection() as connection:
            row = connection.execute(
                "SELECT d.page_count FROM documents d JOIN document_processing p "
                "ON p.document_hash = d.content_hash WHERE d.content_hash = ? "
                "AND d.source_path = ? AND d.size_bytes = ? AND p.processing_key = ?",
                (document.content_hash, str(document.path.resolve()), document.size_bytes, processing_key),
            ).fetchone()
            if row is None:
                return None
            qualities = connection.execute(
                "SELECT text_quality, COUNT(*) FROM pages WHERE document_hash = ? GROUP BY text_quality",
                (document.content_hash,),
            ).fetchall()
            if sum(count for _, count in qualities) != row[0] or not row[0]:
                return None
            incomplete = connection.execute(
                "SELECT COUNT(*) FROM pages p WHERE p.document_hash = ? "
                "AND LENGTH(TRIM(p.normalized_text)) > 0 AND ("
                "NOT EXISTS (SELECT 1 FROM chunks c WHERE c.document_hash = p.document_hash "
                "AND c.page_number = p.page_number) OR "
                "NOT EXISTS (SELECT 1 FROM page_blocks b WHERE b.document_hash = p.document_hash "
                "AND b.page_number = p.page_number))",
                (document.content_hash,),
            ).fetchone()[0]
            if incomplete:
                return None
            chunks = connection.execute(
                "SELECT COUNT(*) FROM chunks WHERE document_hash = ?",
                (document.content_hash,),
            ).fetchone()[0]
            visuals = connection.execute(
                "SELECT COUNT(*) FROM pages WHERE document_hash = ? AND visually_important = 1",
                (document.content_hash,),
            ).fetchone()[0]
            return CachedDocument(
                row[0],
                chunks,
                visuals,
                {PageTextQuality(quality): count for quality, count in qualities},
            )

    def remove_missing_documents(self, paths: set[str]) -> None:
        """Remove index rows for sources absent from the selected library."""
        with self._connection() as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            removed = [
                (content_hash,)
                for content_hash, path in connection.execute(
                    "SELECT content_hash, source_path FROM documents"
                )
                if path not in paths
            ]
            connection.executemany("DELETE FROM documents WHERE content_hash = ?", removed)
            connection.execute("DELETE FROM chunks_fts")
            connection.execute("INSERT INTO chunks_fts(chunk_id, text) SELECT chunk_id, text FROM chunks")

    def counts(self) -> tuple[int, int, int]:
        with self._connection() as connection:
            return tuple(
                connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("documents", "pages", "chunks")
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _ensure_page_columns(connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(pages)").fetchall()}
        migrations = {
            "ocr_text": "ALTER TABLE pages ADD COLUMN ocr_text TEXT",
            "text_source": ("ALTER TABLE pages ADD COLUMN text_source TEXT NOT NULL DEFAULT 'native'"),
            "ocr_engine": "ALTER TABLE pages ADD COLUMN ocr_engine TEXT",
        }
        for column, statement in migrations.items():
            if column not in existing:
                connection.execute(statement)

    @staticmethod
    def _ensure_chunk_columns(connection: sqlite3.Connection) -> None:
        existing = {row[1] for row in connection.execute("PRAGMA table_info(chunks)").fetchall()}
        if "region_index" not in existing:
            connection.execute("ALTER TABLE chunks ADD COLUMN region_index INTEGER NOT NULL DEFAULT 0")
