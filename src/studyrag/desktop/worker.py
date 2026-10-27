"""One study request per child process; stdin/stdout carry JSON, never shell code."""

from collections.abc import Callable
import contextlib
import json
from pathlib import Path
import sys

from studyrag.retrieval import E5Embedder
from studyrag.settings import DEFAULT_EMBEDDING_MODEL, DEFAULT_GENERATION_MODEL, OLLAMA_BASE_URL
from studyrag.study import OllamaGenerator, StudyMode, stream_answer_question
from studyrag.study.conversations import bounded_history, MAX_QUESTION_CHARACTERS


def run_request(
    request: dict,
    emit: Callable[[dict], None],
    *,
    embedder_factory=E5Embedder,
    generator_factory=OllamaGenerator,
) -> None:
    database = Path(request["database"]).expanduser().resolve()
    if not database.is_file():
        raise FileNotFoundError("The library database does not exist. Update the library first.")
    question = request["prompt"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Enter a question or topic first.")
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ValueError(f"Keep each question within {MAX_QUESTION_CHARACTERS} characters.")
    history = bounded_history(request.get("history", ()))
    mode = StudyMode(request.get("mode", "answer"))
    emit({"type": "status", "message": "Loading the local retrieval model..."})
    embedder = embedder_factory(DEFAULT_EMBEDDING_MODEL, local_files_only=True)
    emit({"type": "model", "message": "Retrieving sources and generating an answer..."})
    generator = generator_factory(DEFAULT_GENERATION_MODEL, OLLAMA_BASE_URL)
    for item in stream_answer_question(database, embedder, generator, question, mode=mode, history=history):
        if isinstance(item, str):
            emit({"type": "token", "text": item})
        else:
            emit(
                {
                    "type": "done",
                    "answer": item.text,
                    "topic": item.topic,
                    "sources": [
                        {
                            "id": source.source_id,
                            "file": source.source_path.name,
                            "path": str(source.source_path),
                            "page": source.page_number,
                        }
                        for source in item.sources
                    ],
                }
            )


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    output = sys.stdout

    def emit(event):
        output.write(json.dumps(event, ensure_ascii=False) + "\n")
        output.flush()

    try:
        request = json.loads(sys.stdin.readline())
        if not isinstance(request, dict):
            raise ValueError("Invalid study request")
        # Model-library progress belongs on stderr, not in the JSON event stream.
        with contextlib.redirect_stdout(sys.stderr):
            run_request(request, emit)
    except Exception as error:
        emit({"type": "error", "message": str(error)})
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
