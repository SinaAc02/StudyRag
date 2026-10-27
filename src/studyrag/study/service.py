import re
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from collections.abc import Iterator

from studyrag.retrieval.embeddings import Embedder
from studyrag.retrieval.service import SearchResult, search_index
from studyrag.study.generation import StreamingTextGenerator, TextGenerator
from studyrag.study.conversations import bounded_history, MAX_QUESTION_CHARACTERS


_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")
_TEACHING_REQUEST_PATTERN = re.compile(
    r"\b(?:teach(?:\s+me)?|explain|help\s+me\s+understand|learn\s+about)\b"
    r"|(?:آموزش|توضیح|یاد\s+بده|شرح\s+بده)"
    r"|\b(?:erkl[aä]r|beibringen|lehr)\w*\b",
    flags=re.IGNORECASE,
)


class StudyMode(StrEnum):
    ANSWER = "answer"
    EXPLAIN = "explain"
    COMPARE = "compare"
    QUIZ = "quiz"


_MODE_INSTRUCTIONS = {
    StudyMode.ANSWER: "Answer the question directly and concisely.",
    StudyMode.EXPLAIN: (
        "Teach the topic step by step. Define important terms and include a simple example "
        "when the sources support one."
    ),
    StudyMode.COMPARE: (
        "Compare the requested concepts using clear similarities and differences. "
        "Do not add a difference that the sources do not support."
    ),
    StudyMode.QUIZ: (
        "Create five study questions about the requested topic. Mix recall and application. "
        "After the questions, provide a clearly separated answer key. Cite every answer."
    ),
}


@dataclass(frozen=True, slots=True)
class AnswerSource:
    source_id: str
    source_path: Path
    page_number: int
    score: float


@dataclass(frozen=True, slots=True)
class StudyAnswer:
    text: str
    sources: tuple[AnswerSource, ...]
    topic: str = ""


def _build_prompt(
    question: str,
    results: tuple[SearchResult, ...],
    mode: StudyMode,
) -> str:
    contexts = []
    for number, result in enumerate(results, start=1):
        contexts.append(
            f"[S{number}] File: {result.source_path.name}; page: {result.page_number}\n{result.text.strip()}"
        )
    joined_context = "\n\n".join(contexts)
    return f"""You are a warm, clear teacher helping a student learn from their own material.
Answer in the same language as the question and begin directly with the topic.
Use only relevant information from the supplied sources. Silently ignore unrelated passages.
Do not discuss retrieval quality, irrelevant sources, or what other sources contain.
If the relevant evidence is genuinely insufficient, state only the specific missing point; do not open with a generic limitation disclaimer.
For broad teaching requests, build a coherent lesson with a short introduction, logical Markdown headings, concise paragraphs, and useful bullet points.
Define important terms before using them, connect related facts, and include a simple supported example when possible.
Write formulas as readable plain text or Unicode, not raw LaTeX delimiters or dollar signs.
Cite supported factual claims with source markers such as [S1]. Cite only relevant sources and use only marker IDs shown below.
Do not create a bibliography; the application renders the cited source list.

Study mode: {mode.value}
Mode instructions: {_MODE_INSTRUCTIONS[mode]}

Question:
{question.strip()}

Sources:
{joined_context}

Answer:"""


def _effective_mode(question: str, mode: StudyMode) -> StudyMode:
    if mode is StudyMode.ANSWER and _TEACHING_REQUEST_PATTERN.search(question):
        return StudyMode.EXPLAIN
    return mode


def _validate_citations(answer: str, source_count: int) -> tuple[int, ...]:
    citations = tuple(dict.fromkeys(int(value) for value in _CITATION_PATTERN.findall(answer)))
    if not citations:
        raise RuntimeError("The generated answer has no source citations")
    invalid = [number for number in citations if number < 1 or number > source_count]
    if invalid:
        rendered = ", ".join(f"S{number}" for number in invalid)
        raise RuntimeError(f"The generated answer used unknown source citations: {rendered}")
    return citations


def answer_question(
    database_path: Path,
    embedder: Embedder,
    generator: TextGenerator,
    question: str,
    *,
    top_k: int = 5,
    mode: StudyMode = StudyMode.ANSWER,
) -> StudyAnswer:
    prompt, results = _prepare_question(database_path, embedder, question, top_k, mode)
    return _answer_from_text(generator.generate(prompt).strip(), results)


def _prepare_question(
    database_path: Path,
    embedder: Embedder,
    question: str,
    top_k: int,
    mode: StudyMode,
    history: tuple[dict[str, str], ...] = (),
) -> tuple[str, tuple[SearchResult, ...]]:
    if not question.strip():
        raise ValueError("Question must not be empty")
    if len(question) > MAX_QUESTION_CHARACTERS:
        raise ValueError(f"Keep each question within {MAX_QUESTION_CHARACTERS} characters.")
    effective_mode = _effective_mode(question, mode)
    retrieval_top_k = max(top_k, 8) if effective_mode is StudyMode.EXPLAIN else top_k
    topic = conversation_topic(question, history)
    query = question if topic == question else f"{topic}\nFollow-up: {question}"
    results = search_index(database_path, embedder, query, top_k=retrieval_top_k)
    if not results:
        raise RuntimeError("No source chunks were retrieved")
    return _build_prompt(question, results, effective_mode), results


def _answer_from_text(answer: str, results: tuple[SearchResult, ...], topic: str = "") -> StudyAnswer:
    cited_numbers = _validate_citations(answer, len(results))
    sources = tuple(
        AnswerSource(
            source_id=f"S{number}",
            source_path=results[number - 1].source_path,
            page_number=results[number - 1].page_number,
            score=results[number - 1].score,
        )
        for number in cited_numbers
    )
    return StudyAnswer(answer, sources, topic)


_FOLLOW_UP = re.compile(
    r"\b(?:it|its|this|that|these|those|they|them|simpler|continue)\b"
    r"|\bgive (?:me )?(?:an?|another) example\b|مثال بزن|ساده‌تر|بیشتر توضیح|ادامه بده"
    r"|^(?:give (?:me |an? )?|what about|how about|why\b|and\b|more\b|quiz me|"
    r"advantages\b|disadvantages\b|gib\b|und\b|warum\b|weiter\b|einfacher\b|"
    r"مثال|بیشتر|ساده|چرا|ادامه|تفاوتش)",
    re.IGNORECASE,
)


def conversation_topic(question: str, history: tuple[dict[str, str], ...]) -> str:
    if history and _FOLLOW_UP.search(question.strip()):
        return history[-1].get("topic", history[-1]["question"])[:2000]
    return question


def _chat_messages(prompt: str, history: tuple[dict[str, str], ...]) -> list[dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "Continue the study conversation. Previous messages provide conversational context only, "
                "not authoritative evidence. Prior answers can contain mistakes. Support factual claims "
                "using only the fresh Sources in the current request. Source IDs belong to this turn only; "
                "never reuse citations from earlier answers. Follow short-answer requests."
            ),
        }
    ]
    for turn in history:
        messages.append({"role": "user", "content": turn["question"]})
        messages.append({"role": "assistant", "content": _CITATION_PATTERN.sub("", turn["answer"])})
    messages.append({"role": "user", "content": prompt})
    return messages


def stream_answer_question(
    database_path: Path,
    embedder: Embedder,
    generator: StreamingTextGenerator,
    question: str,
    *,
    top_k: int = 5,
    mode: StudyMode = StudyMode.ANSWER,
    history: tuple[dict[str, str], ...] = (),
) -> Iterator[str | StudyAnswer]:
    """Yield tentative text, followed by a citation-validated final answer."""
    history = bounded_history(history)
    prompt, results = _prepare_question(database_path, embedder, question, top_k, mode, history)
    # Keep source evidence bounded as conversation context grows. Citation indices
    # must refer only to passages actually supplied in this turn.
    budget = 10000
    selected = []
    for result in results:
        if budget <= 0:
            break
        text = result.text[: min(2000, budget)]
        selected.append(replace(result, text=text))
        budget -= len(text)
    results = tuple(selected)
    prompt = _build_prompt(question, results, _effective_mode(question, mode))
    if history:
        prompt = f"Conversation topic: {conversation_topic(question, history)}\n\n{prompt}"
    fragments = []
    if callable(getattr(generator, "stream_chat", None)):
        stream = generator.stream_chat(_chat_messages(prompt, history))
    else:
        # Preserve compatibility with text-generator integrations and CLI fixtures.
        if history:
            prompt = (
                "Conversation context (not source evidence):\n"
                + json.dumps(
                    _chat_messages("", history)[1:-1],
                    ensure_ascii=False,
                )
                + "\n\n"
                + prompt
            )
        stream = generator.stream(prompt)
    for fragment in stream:
        fragments.append(fragment)
        yield fragment
    yield _answer_from_text("".join(fragments).strip(), results, conversation_topic(question, history))
