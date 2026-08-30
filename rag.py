import logging
from pathlib import Path
from uuid import uuid4

import httpx
import pymupdf
from qdrant_client import QdrantClient, models

from models import Answer, Chunk, IndexStatus, Source


LOGGER = logging.getLogger(__name__)
COLLECTION = "study_documents"


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: list[str] = []
    start = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunks.append(clean[start:end].strip())
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


class StudyRAG:
    def __init__(
        self,
        documents_dir: Path,
        ollama_url: str = "http://127.0.0.1:11434",
        ollama_model: str = "gemma3:1b",
        embedding_model: str = "BAAI/bge-m3",
    ) -> None:
        self.documents_dir = documents_dir
        self.ollama_url = ollama_url.rstrip("/")
        self.ollama_model = ollama_model
        self.embedding_model = embedding_model
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(":memory:")
        self._embedder = None
        self.status = IndexStatus(state="starting", documents=[], chunks=0)

    def load_embedding_model(self):
        if self._embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            LOGGER.info("Loading %s on %s", self.embedding_model, device)
            self._embedder = SentenceTransformer(self.embedding_model, device=device)
        return self._embedder

    def index_documents(self) -> IndexStatus:
        self.status = IndexStatus(
            state="indexing",
            documents=self.get_pdf_names(),
            chunks=self.get_chunk_count(),
            message="Reading the documents folder",
        )

        try:
            pdf_files = self.get_pdf_files()

            if self.client.collection_exists(COLLECTION):
                self.client.delete_collection(COLLECTION)

            chunks = []
            for pdf_file in pdf_files:
                file_chunks = self.read_pdf(pdf_file)
                chunks.extend(file_chunks)

            if chunks:
                texts = []
                for chunk in chunks:
                    texts.append(chunk.text)

                embedding_model = self.load_embedding_model()
                vectors = embedding_model.encode(
                    texts,
                    batch_size=16,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                ).tolist()

                self.create_collection_if_needed(len(vectors[0]))
                points = []
                for index in range(len(chunks)):
                    points.append(
                        models.PointStruct(
                            id=chunks[index].chunk_id,
                            vector=vectors[index],
                            payload=chunks[index].model_dump(),
                        )
                    )
                self.client.upsert(
                    collection_name=COLLECTION,
                    wait=True,
                    points=points,
                )

            self.status = IndexStatus(
                state="ready",
                documents=self.get_pdf_names(),
                chunks=self.get_chunk_count(),
                message=f"Indexed {len(pdf_files)} document(s)",
            )
        except Exception as error:
            LOGGER.exception("Document indexing failed")
            self.status = IndexStatus(
                state="error",
                documents=self.get_pdf_names(),
                chunks=self.get_chunk_count(),
                message=str(error),
            )

        return self.status

    def ask(self, question: str, limit: int = 8) -> Answer:
        if not self.client.collection_exists(COLLECTION):
            raise RuntimeError("No indexed PDFs. Put PDF files in documents and reindex.")
        embedding_model = self.load_embedding_model()
        query_vector = embedding_model.encode(
            question,
            normalize_embeddings=True,
        ).tolist()
        result = self.client.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True,
        ).points
        chunks = []
        for point in result:
            if point.payload:
                chunks.append(Chunk.model_validate(point.payload))
        if not chunks:
            raise RuntimeError("No relevant PDF text was found.")

        context_parts = []
        for index, chunk in enumerate(chunks, start=1):
            context_parts.append(
                f"[{index}] {chunk.document}, page {chunk.page}\n{chunk.text}"
            )
        context = "\n\n".join(context_parts)
        prompt = (
            "Answer the student's question using only the PDF excerpts below. "
            "If they do not contain the answer, say that clearly. Explain in a helpful "
            "study style and cite evidence inline like [1] or [2].\n\n"
            f"Question: {question}\n\nPDF excerpts:\n{context}"
        )
        try:
            response = httpx.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.ollama_model,
                    "stream": False,
                    "messages": [{"role": "user", "content": prompt}],
                    "options": {"temperature": 0.2},
                },
                timeout=180,
            )
            response.raise_for_status()
            text = response.json()["message"]["content"].strip()
        except (httpx.HTTPError, KeyError, TypeError) as exc:
            raise RuntimeError(
                f"Ollama is unavailable. Make sure it is running with {self.ollama_model}."
            ) from exc

        pages_already_added = []
        sources = []
        for chunk in chunks:
            page_name = chunk.document + " page " + str(chunk.page)
            if page_name not in pages_already_added:
                pages_already_added.append(page_name)
                sources.append(
                    Source(
                        document=chunk.document,
                        page=chunk.page,
                        excerpt=chunk.text[:240],
                    )
                )
        return Answer(text=text, sources=sources)

    def close(self) -> None:
        self.client.close()

    def read_pdf(self, path: Path) -> list[Chunk]:
        chunks: list[Chunk] = []
        try:
            with pymupdf.open(path) as document:
                for page_number, page in enumerate(document, start=1):
                    for text in split_text(page.get_text()):
                        chunk_id = str(uuid4())
                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                document=path.name,
                                page=page_number,
                                text=text,
                            )
                        )
        except Exception as exc:
            raise RuntimeError(f"Could not read {path.name}: {exc}") from exc
        return chunks

    def create_collection_if_needed(self, vector_size: int) -> None:
        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def get_chunk_count(self) -> int:
        if not self.client.collection_exists(COLLECTION):
            return 0
        return int(self.client.count(COLLECTION, exact=True).count)

    def get_pdf_files(self) -> list[Path]:
        pdf_files = []
        for path in self.documents_dir.iterdir():
            if path.is_file() and path.suffix.lower() == ".pdf":
                pdf_files.append(path)
        pdf_files.sort(key=lambda path: path.name.lower())
        return pdf_files

    def get_pdf_names(self) -> list[str]:
        names = []
        for path in self.get_pdf_files():
            names.append(path.name)
        return names
