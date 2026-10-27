"""Local conversation persistence, independent of the replaceable document index."""

from contextlib import closing
from dataclasses import dataclass
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


MAX_HISTORY_TURNS = 4
MAX_HISTORY_CHARACTERS = 10000
MAX_QUESTION_CHARACTERS = 4000


def bounded_history(value) -> tuple[dict[str, str], ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("Conversation history must be a list of completed turns.")
    turns = []
    remaining = MAX_HISTORY_CHARACTERS
    for item in reversed(value[-MAX_HISTORY_TURNS:]):
        if not isinstance(item, dict) or any(
            not isinstance(item.get(key), str) for key in ("question", "answer")
        ):
            raise ValueError("Conversation history contains an invalid turn.")
        topic = item.get("topic", item["question"])
        if not isinstance(topic, str):
            raise ValueError("Conversation topic must be text.")
        topic = topic or item["question"]
        turn = {"question": item["question"][:2000], "answer": item["answer"][:5000], "topic": topic[:2000]}
        size = sum(map(len, turn.values()))
        if size > remaining:
            if turns:
                break
            # Reserve space for the question and topic before trimming a long answer.
            allowance = max(0, remaining - len(turn["question"]) - len(turn["topic"]))
            turn["answer"] = turn["answer"][:allowance]
            size = sum(map(len, turn.values()))
        turns.append(turn)
        remaining -= size
    return tuple(reversed(turns))


@dataclass(frozen=True, slots=True)
class SavedTurn:
    question: str
    answer: str
    mode: str
    sources: tuple[dict, ...]
    topic: str


class ConversationStore:
    def __init__(self, path: Path):
        self.path = path.expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, scope TEXT NOT NULL, title TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL,
                    question TEXT NOT NULL, answer TEXT NOT NULL, mode TEXT NOT NULL,
                    sources TEXT NOT NULL, topic TEXT NOT NULL,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                );
                CREATE INDEX IF NOT EXISTS conversation_turns_chat
                    ON conversation_turns(conversation_id, id);
            """)

    def create(self, scope: str) -> str:
        identifier = uuid4().hex
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "INSERT INTO conversations(id,scope,title) VALUES(?,?,?)", (identifier, scope, "New chat")
            )
        return identifier

    def list(self, scope: str) -> list[tuple[str, str]]:
        with closing(sqlite3.connect(self.path)) as connection:
            return connection.execute(
                "SELECT id,title FROM conversations WHERE scope=? ORDER BY updated_at DESC,rowid DESC",
                (scope,),
            ).fetchall()

    def turns(self, identifier: str) -> tuple[SavedTurn, ...]:
        with closing(sqlite3.connect(self.path)) as connection:
            rows = connection.execute(
                "SELECT question,answer,mode,sources,topic FROM conversation_turns "
                "WHERE conversation_id=? ORDER BY id",
                (identifier,),
            ).fetchall()
        return tuple(
            SavedTurn(q, a, mode, tuple(json.loads(sources)), topic) for q, a, mode, sources, topic in rows
        )

    def append(self, identifier: str, turn: SavedTurn) -> None:
        if not turn.question.strip() or not turn.answer.strip():
            raise ValueError("Only completed, nonempty conversation turns can be saved.")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            if not connection.execute("SELECT 1 FROM conversations WHERE id=?", (identifier,)).fetchone():
                raise ValueError("The selected conversation no longer exists.")
            first = not connection.execute(
                "SELECT 1 FROM conversation_turns WHERE conversation_id=? LIMIT 1",
                (identifier,),
            ).fetchone()
            connection.execute(
                "INSERT INTO conversation_turns(conversation_id,question,answer,mode,sources,topic) "
                "VALUES(?,?,?,?,?,?)",
                (
                    identifier,
                    turn.question,
                    turn.answer,
                    turn.mode,
                    json.dumps(turn.sources, ensure_ascii=False),
                    turn.topic,
                ),
            )
            title = " ".join(turn.question.split())[:70]
            connection.execute(
                "UPDATE conversations SET updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
                "title=CASE WHEN ? THEN ? ELSE title END WHERE id=?",
                (first, title, identifier),
            )
