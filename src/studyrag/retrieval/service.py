from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path

import numpy as np

from studyrag.retrieval.embeddings import Embedder
from studyrag.retrieval.store import RetrievalStore
from studyrag.text.normalization import normalize_for_search


@dataclass(frozen=True, slots=True)
class IndexSummary:
    model_name: str
    chunk_count: int
    dimensions: int
    reused_chunk_count: int = 0


@dataclass(frozen=True, slots=True)
class SearchResult:
    score: float
    chunk_id: str
    text: str
    source_path: Path
    page_number: int


def build_index(
    database_path: Path,
    embedder: Embedder,
    progress: Callable[[str], None] | None = None,
    batch_size: int = 32,
) -> IndexSummary:
    store = RetrievalStore(database_path)
    store.initialize()
    chunks = store.chunks()
    if not chunks:
        raise RuntimeError("No chunks found. Run the ingest command first.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    cached = {item.chunk.chunk_id: item.vector for item in store.embeddings(embedder.model_name)}
    missing = [chunk for chunk in chunks if chunk.chunk_id not in cached]
    if cached and progress is not None:
        progress(f"Reusing {len(cached)} cached embedding vector(s).")
    for start in range(0, len(missing), batch_size):
        batch = missing[start : start + batch_size]
        vectors = embedder.encode_passages([chunk.text for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError("Embedding model returned an unexpected number of vectors")
        cached.update((chunk.chunk_id, vector) for chunk, vector in zip(batch, vectors, strict=True))
        if progress is not None:
            progress(f"Embedded {len(cached)}/{len(chunks)} chunk(s) with {embedder.model_name}.")
    vectors = np.stack([cached[chunk.chunk_id] for chunk in chunks])
    if missing:
        if progress is not None:
            progress(f"Saving {len(chunks)} embedding vector(s) to SQLite...")
        store.replace_embeddings(embedder.model_name, [chunk.chunk_id for chunk in chunks], vectors)
    summary = IndexSummary(
        embedder.model_name,
        len(chunks),
        int(vectors.shape[1]),
        len(chunks) - len(missing),
    )
    if progress is not None:
        progress(f"Index complete: {summary.chunk_count} chunk(s), {summary.dimensions} dimensions.")
    return summary


def search_index(
    database_path: Path,
    embedder: Embedder,
    query: str,
    *,
    top_k: int = 5,
) -> tuple[SearchResult, ...]:
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    store = RetrievalStore(database_path)
    store.initialize()
    stored = store.embeddings(embedder.model_name)
    if not stored:
        raise RuntimeError(f"No index found for model: {embedder.model_name}")
    normalized_query = normalize_for_search(query)
    query_vector = embedder.encode_query(normalized_query)
    matrix = np.stack([item.vector for item in stored])
    if matrix.shape[1] != query_vector.shape[0]:
        raise RuntimeError("Stored embedding dimensions do not match the query model")
    scores = matrix @ query_vector
    candidate_limit = min(len(stored), max(top_k * 4, 20))
    dense_order = np.argsort(scores)[::-1][:candidate_limit]
    keyword_ids = store.keyword_chunk_ids(normalized_query, candidate_limit)

    # Reciprocal Rank Fusion combines semantic and exact-term evidence without
    # requiring their unrelated score scales to be calibrated.
    fused: dict[str, float] = {}
    stored_by_id = {item.chunk.chunk_id: item for item in stored}
    for rank, index in enumerate(dense_order, start=1):
        chunk_id = stored[index].chunk.chunk_id
        fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank)
    for rank, chunk_id in enumerate(keyword_ids, start=1):
        if chunk_id in stored_by_id:
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank)

    ranked_ids = sorted(fused, key=fused.__getitem__, reverse=True)
    max_score = fused[ranked_ids[0]] if ranked_ids else 1.0
    results: list[SearchResult] = []
    selected_regions: set[tuple[str, int, int]] = set()
    for chunk_id in ranked_ids:
        chunk = stored_by_id[chunk_id].chunk
        region_key = (chunk.document_hash, chunk.page_number, chunk.region_index)
        if region_key in selected_regions:
            continue
        selected_regions.add(region_key)
        results.append(
            SearchResult(
                score=fused[chunk_id] / max_score,
                chunk_id=chunk.chunk_id,
                text=store.expanded_region_text(chunk),
                source_path=chunk.source_path,
                page_number=chunk.page_number,
            )
        )
        if len(results) == top_k:
            break
    return tuple(results)
