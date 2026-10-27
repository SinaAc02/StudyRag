from pathlib import Path

import numpy as np
import pytest

from studyrag.domain.models import (
    ExtractedPage,
    ExtractionResult,
    PageAssessment,
    PageTextQuality,
    SourceDocument,
)
from studyrag.ingestion.chunking import chunk_page
from studyrag.ingestion.store import IngestionStore
from studyrag.retrieval.service import build_index
from studyrag.study.service import StudyMode, answer_question


class FakeEmbedder:
    model_name = "fake"

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return np.array([[1.0, 0.0] for _ in texts], dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0], dtype=np.float32)


class FakeGenerator:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompt = ""

    def generate(self, prompt: str) -> str:
        self.prompt = prompt
        return self.answer


def _indexed_database(tmp_path: Path) -> Path:
    database = tmp_path / "study.sqlite"
    path = tmp_path / "lesson.pdf"
    text = "Pearson correlation measures linear association."
    document = SourceDocument(path, "hash", len(text))
    page = ExtractedPage(path, 3, text, text)
    extraction = ExtractionResult(document, (page,))
    assessment = PageAssessment(PageTextQuality.GOOD_NATIVE, len(text), False, ())
    store = IngestionStore(database)
    store.initialize()
    store.replace_document(extraction, (assessment,), chunk_page(document, page))
    build_index(database, FakeEmbedder())
    return database


def test_answer_uses_retrieved_context_and_maps_citations(tmp_path: Path) -> None:
    database = _indexed_database(tmp_path)
    generator = FakeGenerator("Correlation measures linear association [S1].")

    result = answer_question(database, FakeEmbedder(), generator, "What is correlation?")

    assert "Pearson correlation" in generator.prompt
    assert "same language" in generator.prompt
    assert result.sources[0].source_id == "S1"
    assert result.sources[0].source_path.name == "lesson.pdf"
    assert result.sources[0].page_number == 3


@pytest.mark.parametrize("answer", ["Unsupported answer.", "Claim [S2]."])
def test_answer_rejects_missing_or_unknown_citations(tmp_path: Path, answer: str) -> None:
    database = _indexed_database(tmp_path)

    with pytest.raises(RuntimeError):
        answer_question(database, FakeEmbedder(), FakeGenerator(answer), "Question")


@pytest.mark.parametrize(
    ("mode", "expected_instruction"),
    [
        (StudyMode.ANSWER, "Answer the question directly"),
        (StudyMode.EXPLAIN, "Teach the topic step by step"),
        (StudyMode.COMPARE, "Compare the requested concepts"),
        (StudyMode.QUIZ, "Create five study questions"),
    ],
)
def test_study_mode_adds_bounded_instruction(
    tmp_path: Path,
    mode: StudyMode,
    expected_instruction: str,
) -> None:
    database = _indexed_database(tmp_path)
    generator = FakeGenerator("Grounded result [S1].")

    answer_question(database, FakeEmbedder(), generator, "Topic", mode=mode)

    assert f"Study mode: {mode.value}" in generator.prompt
    assert expected_instruction in generator.prompt


@pytest.mark.parametrize(
    "question",
    ["Teach me about fishes", "ماهی‌ها را به من توضیح بده", "Erkläre mir Fische"],
)
def test_teaching_request_uses_explain_behavior_in_answer_mode(
    tmp_path: Path,
    question: str,
) -> None:
    database = _indexed_database(tmp_path)
    generator = FakeGenerator("Grounded lesson [S1].")

    answer_question(database, FakeEmbedder(), generator, question)

    assert "Study mode: explain" in generator.prompt
    assert "Teach the topic step by step" in generator.prompt
    assert "Silently ignore unrelated passages" in generator.prompt
    assert "Do not discuss retrieval quality" in generator.prompt


def test_direct_question_keeps_answer_behavior(tmp_path: Path) -> None:
    database = _indexed_database(tmp_path)
    generator = FakeGenerator("Grounded answer [S1].")

    answer_question(database, FakeEmbedder(), generator, "What is correlation?")

    assert "Study mode: answer" in generator.prompt
