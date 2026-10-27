from pathlib import Path

import pytest

from studyrag.study.conversations import (
    ConversationStore,
    SavedTurn,
    bounded_history,
    MAX_HISTORY_CHARACTERS,
    MAX_HISTORY_TURNS,
)
from studyrag.study.service import stream_answer_question, StudyAnswer
from studyrag.retrieval.service import SearchResult


def saved_turn(question="Explain sampling bias", answer="An explanation [S1]", topic="sampling bias"):
    return SavedTurn(
        question,
        answer,
        "answer",
        ({"id": "S1", "file": "lesson.pdf", "path": "lesson.pdf", "page": 10},),
        topic,
    )


def test_completed_conversations_survive_restart_and_are_separate_from_index(tmp_path):
    index = tmp_path / "index.sqlite"
    index.write_bytes(b"Original index")
    store = ConversationStore(tmp_path / "chats.sqlite")
    identifier = store.create("library-a")
    store.append(identifier, saved_turn())
    restored = ConversationStore(store.path)
    assert restored.turns(identifier) == (saved_turn(),)
    assert restored.list("library-a") == [(identifier, "Explain sampling bias")]
    assert restored.list("library-b") == []
    assert index.read_bytes() == b"Original index"
    # Replacing the document index cannot replace the independent chat database.
    index.write_bytes(b"Rebuilt index")
    assert restored.turns(identifier)[0].sources[0]["page"] == 10


def test_new_chat_preserves_the_old_conversation(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite")
    old = store.create("library")
    store.append(old, saved_turn())
    new = store.create("library")
    assert store.turns(new) == ()
    assert len(store.turns(old)) == 1
    assert len(store.list("library")) == 2


def test_store_rejects_incomplete_turns_and_unknown_chats(tmp_path):
    store = ConversationStore(tmp_path / "chats.sqlite")
    identifier = store.create("library")
    with pytest.raises(ValueError, match="completed"):
        store.append(identifier, saved_turn(answer=""))
    assert store.turns(identifier) == ()
    with pytest.raises(ValueError, match="no longer exists"):
        store.append("missing", saved_turn())


def test_history_is_bounded_and_keeps_newest_complete_pairs():
    history = [{"question": f"Question {n}", "answer": "x" * 4000, "topic": "topic"} for n in range(30)]
    selected = bounded_history(history)
    assert len(selected) <= MAX_HISTORY_TURNS
    assert sum(sum(map(len, item.values())) for item in selected) <= MAX_HISTORY_CHARACTERS
    assert selected[-1]["question"] == "Question 29"
    assert history[-1]["answer"] == "x" * 4000
    huge = bounded_history([{"question": "q" * 10000, "answer": "a" * 50000, "topic": "t" * 10000}])
    assert sum(sum(map(len, item.values())) for item in huge) <= MAX_HISTORY_CHARACTERS


@pytest.mark.parametrize(
    "value",
    ["text", [None], [{"question": "q", "answer": 42}], [{"question": "q", "answer": "a", "topic": []}]],
)
def test_invalid_history_is_rejected(value):
    with pytest.raises(ValueError):
        bounded_history(value)


def test_follow_up_retrieves_fresh_evidence_and_uses_chat_roles(monkeypatch):
    queries = []

    def search(database, embedder, query, top_k):
        queries.append(query)
        page = 10 if len(queries) == 1 else 11
        return (SearchResult(1.0, "chunk", f"Fresh page {page} evidence", Path("lesson.pdf"), page),)

    monkeypatch.setattr("studyrag.study.service.search_index", search)

    class ChatGenerator:
        def __init__(self):
            self.requests = []

        def stream_chat(self, messages):
            self.requests.append(messages)
            yield "Grounded reply [S1]"

    generator = ChatGenerator()
    first = list(stream_answer_question(Path("unused"), None, generator, "Explain sampling bias"))[-1]
    second = list(
        stream_answer_question(
            Path("unused"),
            None,
            generator,
            "Give me an example",
            history=({"question": "Explain sampling bias", "answer": first.text, "topic": first.topic},),
        )
    )[-1]
    assert isinstance(second, StudyAnswer)
    assert "sampling bias" in queries[1] and "Give me an example" in queries[1]
    assert second.sources[0].page_number == 11
    assert second.topic == "Explain sampling bias"
    messages = generator.requests[1]
    assert [message["role"] for message in messages] == ["system", "user", "assistant", "user"]
    assert "[S1]" not in messages[2]["content"]
    assert "Fresh page 11 evidence" in messages[-1]["content"]
    assert "Fresh page 10 evidence" not in messages[-1]["content"]


def test_explicit_new_topic_does_not_retrieve_the_old_topic(monkeypatch):
    queries = []
    monkeypatch.setattr(
        "studyrag.study.service.search_index",
        lambda d, e, q, top_k: (
            queries.append(q) or SearchResult(1, "id", "Evidence", Path("lesson.pdf"), 1),
        ),
    )

    class Generator:
        def stream_chat(self, messages):
            yield "Answer [S1]"

    list(
        stream_answer_question(
            Path("unused"),
            None,
            Generator(),
            "What is photosynthesis?",
            history=(
                {"question": "Explain sampling bias", "answer": "Bias answer [S8]", "topic": "sampling bias"},
            ),
        )
    )
    assert queries == ["What is photosynthesis?"]


def test_followup_cannot_reuse_an_old_citation_id(monkeypatch):
    monkeypatch.setattr(
        "studyrag.study.service.search_index",
        lambda *a, **kw: (SearchResult(1, "id", "New evidence", Path("new.pdf"), 2),),
    )

    class Generator:
        def stream_chat(self, messages):
            assert "[S8]" not in messages[2]["content"]
            yield "Incorrect reused citation [S8]"

    with pytest.raises(RuntimeError, match="unknown source citations"):
        list(
            stream_answer_question(
                Path("unused"),
                None,
                Generator(),
                "Give me an example",
                history=({"question": "Topic", "answer": "Old answer [S8]"},),
            )
        )


@pytest.mark.parametrize(
    "question", ["Can you give me an example?", "Gib mir ein Beispiel", "یک مثال بزن", "بیشتر توضیح بده"]
)
def test_common_multilingual_followups_keep_the_topic(question):
    from studyrag.study.service import conversation_topic

    assert (
        conversation_topic(
            question,
            ({"question": "Earlier question", "answer": "Earlier answer", "topic": "sampling bias"},),
        )
        == "sampling bias"
    )
