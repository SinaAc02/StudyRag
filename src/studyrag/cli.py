import argparse
import sys
from pathlib import Path

from studyrag.ingestion.discovery import discover_documents
from studyrag.ingestion.pipeline import ingest_folder
from studyrag.ingestion.ocr import OcrUnavailableError, PaddleStructureEngine
from studyrag.ingestion.parsers import NativePdfParser
from studyrag.retrieval import E5Embedder, build_index, search_index
from studyrag.study import OllamaGenerator, StudyMode, answer_question
from studyrag.settings import (
    DEFAULT_DATABASE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MATERIAL_FOLDER,
)


def _console_progress(message: str) -> None:
    print(f"[StudyRag] {message}", flush=True)


def _scan(folder: Path) -> int:
    documents = list(discover_documents(folder))
    if not documents:
        print(f"No supported documents found in: {folder.resolve()}")
        return 0

    parser = NativePdfParser()
    print(f"Found {len(documents)} PDF document(s).")
    for document in documents:
        result = parser.parse(document)
        text_pages = sum(bool(page.normalized_text) for page in result.pages)
        status = "native text" if text_pages else "OCR needed"
        print(
            f"- {document.path.name}: {len(result.pages)} page(s), "
            f"{result.character_count} character(s), {status}, "
            f"sha256={document.content_hash[:12]}"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="studyrag")
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan", help="Inspect a learning-material folder")
    scan.add_argument("folder", type=Path)
    ingest = subparsers.add_parser("ingest", help="Extract, assess, chunk, and store documents")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument(
        "--database",
        type=Path,
        default=DEFAULT_DATABASE,
        help="SQLite output path (default: application data directory)",
    )
    index = subparsers.add_parser("index", help="Embed and persist ingested chunks")
    index.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    index.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    search = subparsers.add_parser("search", help="Retrieve source chunks for a question")
    search.add_argument("query")
    search.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    search.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL)
    search.add_argument("--top-k", type=int, default=5)
    ask = subparsers.add_parser("ask", help="Answer from retrieved sources with citations")
    ask.add_argument("question")
    ask.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    ask.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ask.add_argument(
        "--model",
        default=DEFAULT_GENERATION_MODEL,
        help=f"Installed Ollama model (default: {DEFAULT_GENERATION_MODEL})",
    )
    ask.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument(
        "--mode",
        choices=tuple(mode.value for mode in StudyMode),
        default=StudyMode.ANSWER.value,
        help="Study behavior: answer, explain, compare, or quiz",
    )
    desktop = subparsers.add_parser("desktop", help="Open the native StudyRag desktop app")
    desktop.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    desktop.add_argument("--material", type=Path, default=DEFAULT_MATERIAL_FOLDER)
    return parser


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    arguments = build_parser().parse_args(None if len(sys.argv) > 1 else ["desktop"])
    if arguments.command == "scan":
        return _scan(arguments.folder)
    if arguments.command == "ingest":
        try:
            summary = ingest_folder(
                arguments.folder,
                arguments.database,
                ocr_engine=PaddleStructureEngine(),
                progress=_console_progress,
            )
        except OcrUnavailableError as error:
            print(f"OCR unavailable: {error}", file=sys.stderr)
            return 1
        print(f"Ingested {summary.document_count} document(s), {summary.page_count} page(s).")
        print(
            "Page quality: "
            f"good={summary.quality_counts.get('good_native', 0)}, "
            f"sparse={summary.quality_counts.get('sparse_native', 0)}, "
            f"ocr_needed={summary.quality_counts.get('ocr_needed', 0)}"
        )
        print(f"Visually important pages: {summary.visual_page_count}")
        print(f"Pages processed with OCR: {summary.ocr_page_count}")
        print(f"Pages using native extraction this run: {summary.native_page_count}")
        print(
            f"Reused unchanged documents: {summary.reused_document_count} "
            f"({summary.reused_page_count} page(s))"
        )
        print(f"Stored chunks: {summary.chunk_count}")
        print(f"Database: {summary.database_path}")
        return 0
    if arguments.command == "index":
        embedder = E5Embedder(arguments.model, local_files_only=True)
        summary = build_index(arguments.database, embedder, _console_progress)
        print(
            f"Indexed {summary.chunk_count} chunk(s) with {summary.model_name} "
            f"({summary.dimensions} dimensions)."
        )
        return 0
    if arguments.command == "search":
        embedder = E5Embedder(arguments.model, local_files_only=True)
        results = search_index(
            arguments.database,
            embedder,
            arguments.query,
            top_k=arguments.top_k,
        )
        for rank, result in enumerate(results, start=1):
            preview = " ".join(result.text.split())[:240]
            print(
                f"{rank}. score={result.score:.4f} source={result.source_path.name} page={result.page_number}"
            )
            print(f"   {preview}")
        return 0
    if arguments.command == "ask":
        embedder = E5Embedder(arguments.embedding_model, local_files_only=True)
        generator = OllamaGenerator(arguments.model, arguments.ollama_url)
        try:
            answer = answer_question(
                arguments.database,
                embedder,
                generator,
                arguments.question,
                top_k=arguments.top_k,
                mode=StudyMode(arguments.mode),
            )
        except RuntimeError as error:
            print(f"Answer failed: {error}", file=sys.stderr)
            return 1
        print(answer.text)
        print("\nSources:")
        for source in answer.sources:
            print(
                f"[{source.source_id}] {source.source_path.name}, "
                f"page {source.page_number} (retrieval score {source.score:.4f})"
            )
        return 0
    if arguments.command == "desktop":
        from studyrag.desktop.app import launch

        return launch(database=arguments.database, material=arguments.material)
    return 2
