from fastapi.testclient import TestClient

from app import create_app
from models import Answer, IndexStatus, Source
from rag import split_text


class FakeRAG:
    def __init__(self) -> None:
        self.status = IndexStatus(state="starting", documents=[], chunks=0)

    def index_documents(self) -> IndexStatus:
        self.status = IndexStatus(
            state="ready",
            documents=["notes.pdf"],
            chunks=3,
            message="Indexed 1 document(s)",
        )
        return self.status

    def ask(self, question: str) -> Answer:
        return Answer(
            text=f"Answer to {question}",
            sources=[Source(document="notes.pdf", page=2, excerpt="Evidence")],
        )

    def close(self) -> None:
        pass


def test_small_text_stays_in_one_chunk() -> None:
    assert split_text("A short page") == ["A short page"]


def test_long_text_is_split_with_overlap() -> None:
    chunks = split_text("word " * 500, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(chunk) <= 100 for chunk in chunks)


def test_api_and_static_page() -> None:
    with TestClient(create_app(FakeRAG())) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/health").json() == {"status": "ok"}
        assert client.get("/api/status").json()["documents"] == ["notes.pdf"]
        response = client.post("/api/explain", json={"text": "Adam"})
        assert response.status_code == 200
        assert response.json()["sources"][0]["page"] == 2


def test_empty_question_is_rejected() -> None:
    with TestClient(create_app(FakeRAG())) as client:
        assert client.post("/api/explain", json={"text": ""}).status_code == 422
        assert client.post("/api/explain", json={"text": "   "}).status_code == 422
