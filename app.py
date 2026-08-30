import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from models import Answer, IndexStatus, Question
from rag import StudyRAG


ROOT = Path(__file__).resolve().parent


def create_app(engine: StudyRAG | None = None) -> FastAPI:
    rag = engine or StudyRAG(
        documents_dir=ROOT / "documents",
        ollama_url=os.getenv("OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "gemma3:1b"),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        rag.index_documents()
        yield
        rag.close()

    application = FastAPI(title="Study RAG", lifespan=lifespan)

    @application.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @application.get("/api/status", response_model=IndexStatus)
    def status() -> IndexStatus:
        return rag.status

    @application.post("/api/reindex", response_model=IndexStatus)
    def reindex() -> IndexStatus:
        return rag.index_documents()

    @application.post("/api/explain", response_model=Answer)
    def explain(question: Question) -> Answer:
        text = question.text.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Enter a question.")
        try:
            return rag.ask(text)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    application.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
    return application


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
app = create_app()
