from io import BytesIO
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Event, Thread
import json

import pytest

from studyrag.retrieval.service import SearchResult
from studyrag.study.generation import OllamaGenerator
from studyrag.study.service import StudyAnswer, stream_answer_question
import studyrag.study.service as study_service


def response_bytes(frames):
    return BytesIO(
        b"".join(json.dumps(frame, ensure_ascii=False).encode("utf-8") + b"\n" for frame in frames)
    )


def test_generate_sends_stream_true_and_assembles_utf8_fragments(monkeypatch):
    recorded = []

    def open_response(request, timeout):
        recorded.append(json.loads(request.data))
        return response_bytes(
            [
                {"thinking": "internal reasoning", "done": False},
                {"response": "پاسخ ", "done": False},
                {"response": "[S1]", "done": True},
            ]
        )

    monkeypatch.setattr("studyrag.study.generation.urlopen", open_response)
    assert OllamaGenerator("local-model").generate("question") == "پاسخ [S1]"
    assert recorded[0]["stream"] is True


@pytest.mark.parametrize(
    "frames,message",
    [
        ([{"response": "partial", "done": False}], "before generation completed"),
        ([{"error": "model unavailable"}], "model unavailable"),
        ([{"response": 123, "done": True}], "invalid answer text"),
    ],
)
def test_protocol_failures_are_explicit(monkeypatch, frames, message):
    monkeypatch.setattr("studyrag.study.generation.urlopen", lambda *a, **k: response_bytes(frames))
    with pytest.raises(RuntimeError, match=message):
        list(OllamaGenerator("local-model").stream("question"))


def test_malformed_stream_is_rejected(monkeypatch):
    monkeypatch.setattr(
        "studyrag.study.generation.urlopen",
        lambda *a, **k: BytesIO(b"not-json\n"),
    )
    with pytest.raises(RuntimeError, match="invalid streaming response"):
        list(OllamaGenerator("local-model").stream("question"))


def test_chat_sends_history_and_streams_only_visible_content(monkeypatch):
    requests = []

    def open_response(request, timeout):
        requests.append((request.full_url, json.loads(request.data)))
        return response_bytes(
            [
                {"message": {"thinking": "internal reasoning"}, "done": False},
                {"message": {"role": "assistant", "content": "مثال "}, "done": False},
                {"message": {"content": "[S1]"}, "done": True},
            ]
        )

    monkeypatch.setattr("studyrag.study.generation.urlopen", open_response)
    messages = [
        {"role": "user", "content": "Topic"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Give me an example"},
    ]
    assert list(OllamaGenerator("local").stream_chat(messages)) == ["مثال ", "[S1]"]
    assert requests[0][0].endswith("/api/chat")
    assert requests[0][1]["messages"] == messages
    assert requests[0][1]["stream"] is True
    assert requests[0][1]["options"]["num_ctx"] == 16384


@pytest.mark.parametrize(
    "frames",
    [
        [{"message": "invalid", "done": True}],
        [{"message": {"content": 42}, "done": True}],
        [{"message": {"content": "partial"}, "done": False}],
    ],
)
def test_chat_rejects_invalid_or_incomplete_streams(monkeypatch, frames):
    monkeypatch.setattr("studyrag.study.generation.urlopen", lambda *a, **k: response_bytes(frames))
    with pytest.raises(RuntimeError):
        list(OllamaGenerator("local").stream_chat([{"role": "user", "content": "question"}]))


def test_shutdown_unloads_only_the_configured_model(monkeypatch):
    requests = []

    def open_response(request, timeout):
        requests.append((json.loads(request.data), timeout))
        return BytesIO(b'{"done":true}')

    monkeypatch.setattr("studyrag.study.generation.urlopen", open_response)
    OllamaGenerator("study-model").unload()
    assert requests == [({"model": "study-model", "keep_alive": 0, "stream": False}, 10)]


def test_real_http_stream_yields_before_server_finishes():
    release = Event()
    finished = Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return None

        def do_POST(self):
            requests.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"response":"First ","done":false}\n')
            self.wfile.flush()
            release.wait(5)
            self.wfile.write(b'{"response":"[S1]","done":true}\n')
            self.wfile.flush()
            finished.set()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    generator = OllamaGenerator("fake-local", f"http://127.0.0.1:{server.server_port}")
    stream = generator.stream("test question")
    try:
        assert next(stream) == "First "
        assert not finished.is_set()
        release.set()
        assert list(stream) == ["[S1]"]
        assert requests[0]["stream"] is True
    finally:
        release.set()
        stream.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_real_chat_http_stream_keeps_history_and_yields_before_completion():
    release = Event()
    recorded = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            recorded.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            self.wfile.write(b'{"message":{"content":"An example "},"done":false}\n')
            self.wfile.flush()
            release.wait(5)
            self.wfile.write(b'{"message":{"content":"[S1]"},"done":true}\n')
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    messages = [
        {"role": "user", "content": "Explain sampling bias"},
        {"role": "assistant", "content": "A prior explanation"},
        {"role": "user", "content": "Give me an example"},
    ]
    stream = OllamaGenerator("fake", f"http://127.0.0.1:{server.server_port}").stream_chat(messages)
    try:
        assert next(stream) == "An example "
        assert recorded == [
            (
                "/api/chat",
                {
                    "model": "fake",
                    "messages": messages,
                    "stream": True,
                    "options": {"temperature": 0.2, "num_ctx": 16384},
                },
            )
        ]
        release.set()
        assert list(stream) == ["[S1]"]
    finally:
        release.set()
        stream.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("final,valid", [("[S1]", True), ("[S9]", False), ("no source", False)])
def test_stream_validates_citations_only_after_tentative_tokens(monkeypatch, final, valid):
    monkeypatch.setattr(
        study_service,
        "search_index",
        lambda *a, **k: (SearchResult(1.0, "chunk", "Evidence", Path("lesson.pdf"), 7),),
    )

    class Generator:
        def stream(self, prompt):
            yield "An explanation "
            yield final

    stream = stream_answer_question(Path("unused.sqlite"), None, Generator(), "question")
    assert next(stream) == "An explanation "
    assert next(stream) == final
    if valid:
        answer = next(stream)
        assert isinstance(answer, StudyAnswer)
        assert answer.sources[0].page_number == 7
    else:
        with pytest.raises(RuntimeError, match="citations"):
            next(stream)
