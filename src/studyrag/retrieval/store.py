import sqlite3
import re
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True, slots=True)
class StoredChunk:
    chunk_id: str
    document_hash: str
    text: str
    source_path: Path
    page_number: int
    chunk_index: int
    region_index: int


@dataclass(frozen=True, slots=True)
class StoredEmbedding:
    chunk: StoredChunk
    vector: np.ndarray


class RetrievalStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path.expanduser().resolve()

    def initialize(self) -> None:
        if not self.database_path.exists():
            raise FileNotFoundError(f"Ingestion database does not exist: {self.database_path}")
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    chunk_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    vector BLOB NOT NULL,
                    PRIMARY KEY (chunk_id, model_name),
                    FOREIGN KEY (chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    text,
                    tokenize = 'unicode61'
                )
                """
            )
            indexed_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
            chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            if indexed_count != chunk_count:
                connection.execute("DELETE FROM chunks_fts")
                connection.execute("INSERT INTO chunks_fts(chunk_id, text) SELECT chunk_id, text FROM chunks")

    def chunks(self) -> list[StoredChunk]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_hash, c.text, d.source_path,
                       c.page_number, c.chunk_index, c.region_index
                FROM chunks c
                JOIN documents d ON d.content_hash = c.document_hash
                ORDER BY d.source_path, c.page_number, c.chunk_index
                """
            ).fetchall()
        return [StoredChunk(row[0], row[1], row[2], Path(row[3]), row[4], row[5], row[6]) for row in rows]

    def replace_embeddings(
        self,
        model_name: str,
        chunk_ids: list[str],
        vectors: np.ndarray,
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("Each chunk requires exactly one embedding")
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a non-empty two-dimensional matrix")
        matrix = np.asarray(vectors, dtype=np.float32)
        dimensions = int(matrix.shape[1])
        with self._connection() as connection:
            connection.execute("DELETE FROM embeddings WHERE model_name = ?", (model_name,))
            connection.executemany(
                "INSERT INTO embeddings(chunk_id, model_name, dimensions, vector) VALUES (?, ?, ?, ?)",
                [
                    (chunk_id, model_name, dimensions, vector.tobytes())
                    for chunk_id, vector in zip(chunk_ids, matrix, strict=True)
                ],
            )

    def embeddings(self, model_name: str) -> list[StoredEmbedding]:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT c.chunk_id, c.document_hash, c.text, d.source_path,
                       c.page_number, c.chunk_index, c.region_index,
                       e.dimensions, e.vector
                FROM embeddings e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                JOIN documents d ON d.content_hash = c.document_hash
                WHERE e.model_name = ?
                ORDER BY d.source_path, c.page_number, c.chunk_index
                """,
                (model_name,),
            ).fetchall()
        return [
            StoredEmbedding(
                StoredChunk(row[0], row[1], row[2], Path(row[3]), row[4], row[5], row[6]),
                np.frombuffer(row[8], dtype=np.float32, count=row[7]).copy(),
            )
            for row in rows
        ]

    def keyword_chunk_ids(self, query: str, limit: int) -> list[str]:
        terms = list(dict.fromkeys(re.findall(r"\w+", query.casefold(), flags=re.UNICODE)))
        if not terms or limit < 1:
            return []
        fts_query = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        return [row[0] for row in rows]

    def expanded_region_text(self, chunk: StoredChunk) -> str:
        with self._connection() as connection:
            rows = connection.execute(
                """
                SELECT text
                FROM chunks
                WHERE document_hash = ? AND page_number = ? AND region_index = ?
                ORDER BY chunk_index
                """,
                (chunk.document_hash, chunk.page_number, chunk.region_index),
            ).fetchall()
        return "\n".join(row[0].strip() for row in rows if row[0].strip())

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            with connection:
                yield connection
        finally:
            connection.close()
