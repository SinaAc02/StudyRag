from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def model_name(self) -> str: ...

    def encode_passages(self, texts: list[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class E5Embedder:
    """Multilingual E5 adapter with the prefixes required by its model card."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        *,
        local_files_only: bool = True,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, local_files_only=local_files_only)

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self.model.encode(
                [f"passage: {text}" for text in texts],
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def encode_query(self, text: str) -> np.ndarray:
        if not text.strip():
            raise ValueError("Search query cannot be empty")
        vectors = self.model.encode(
            [f"query: {text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors[0], dtype=np.float32)
