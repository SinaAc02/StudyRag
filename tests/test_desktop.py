import json
import os
from pathlib import Path
import sys
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import QApplication

from studyrag.desktop import app as desktop
from studyrag.desktop.worker import run_request
from studyrag.cli import build_parser
from studyrag.study.service import AnswerSource, StudyAnswer


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    return app


def wait_for(app, predicate, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Desktop operation did not finish before the test timeout")


class Coordinator:
    _thread = None

    def __init__(self, root):
        self.database_path = root / "missing.sqlite"
        self.material_folder = root
        self.fresh = True
        self.cancelled = False
        self.running = False

    def _is_fresh(self):
        return self.fresh

    def status(self):
        return {"running": self.running, "percent": 50, "message": "Preparing test library", "error": None}

    def start(self, folder, force=False):
        self.material_folder = folder
        self.running = True
        return True

    def cancel(self):
        self.cancelled = True
        self.running = False


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(
        """import json,sys,time
from pathlib import Path
request=json.loads(sys.stdin.readline())
with Path(__file__).with_suffix(".requests.jsonl").open("a",encoding="utf-8") as log:
 log.write(json.dumps(request,ensure_ascii=False)+"\\n")
def emit(event):
 print(json.dumps(event,ensure_ascii=False),flush=True)
emit({"type":"model","message":"Generating test response"})
emit({"type":"token","text":"Partial answer "})
if "hang" in request["prompt"]: time.sleep(60)
time.sleep(0.15)
if "invalid" in request["prompt"]:
 emit({"type":"error","message":"Unknown source citation: S99"})
 sys.exit(1)
text="پاسخ فارسی [S1]" if "persian" in request["prompt"] else "Final answer [S1]"
emit({"type":"done","answer":text,"sources":[{"id":"S1","file":"lesson.pdf","path":request["database"],"page":3}]})
if "badexit" in request["prompt"]: sys.exit(1)
""",
        encoding="utf-8",
    )

    class NoNetworkGenerator:
        def __init__(self, *args):
            pass

        def unload(self):
            pass

    monkeypatch.setattr(desktop, "OllamaGenerator", NoNetworkGenerator)
    coordinator = Coordinator(tmp_path)
    ui = desktop.StudyWindow(
        database=coordinator.database_path,
        material=tmp_path,
        coordinator=coordinator,
        worker_program=(sys.executable, ["-B", str(worker)]),
    )
    ui.show()
    wait_for(application, lambda: ui.checker is None and ui.ready)
    yield ui
    if not ui.close_ready:
        ui.close()
        wait_for(application, lambda: ui.close_ready)
    ui.deleteLater()
    application.processEvents()


def test_cli_has_native_desktop_command_and_no_web_server():
    parser = build_parser()
    assert parser.parse_args(["desktop"]).command == "desktop"
    with pytest.raises(SystemExit):
        parser.parse_args(["serve"])


def test_startup_does_not_start_library_preparation(window):
    assert window.ready and window.ask.isEnabled()
    assert not window.coordinator.running
    assert window.process.state() == QProcess.ProcessState.NotRunning


def test_stream_updates_native_answer_then_shows_sources(application, window):
    window.prompt.setPlainText("question")
    window.ask_question()
    wait_for(application, lambda: "Partial answer" in window.answer.toPlainText())
    assert not window.completed and window.sources.topLevelItemCount() == 0
    assert not window.ask.isEnabled()
    wait_for(application, lambda: not window._busy())
    assert window.completed
    assert "question" in window.answer.toPlainText()
    assert "Final answer [S1]" in window.answer.toPlainText()
    assert window.sources.topLevelItem(0).child(0).text(2) == "3"
    assert window.ask.isEnabled()


def test_failed_answer_clears_draft_and_sources(application, window):
    window.prompt.setPlainText("invalid citation")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    assert window.answer.toPlainText() == ""
    assert window.sources.topLevelItemCount() == 0
    assert "S99" in window.message.text()
    assert window.ask.isEnabled()


def test_cancel_stops_owned_worker_and_restores_controls(application, window):
    window.prompt.setPlainText("hang")
    window.ask_question()
    wait_for(application, lambda: "Partial answer" in window.answer.toPlainText())
    window.cancel_answer()
    wait_for(application, lambda: not window._busy())
    assert window.answer.toPlainText() == ""
    assert "cancelled" in window.message.text()
    assert window.ask.isEnabled()


def test_close_stops_worker_without_browser_heartbeat(application, window):
    window.prompt.setPlainText("hang")
    window.ask_question()
    wait_for(application, lambda: "Partial answer" in window.answer.toPlainText())
    window.close()
    wait_for(application, lambda: window.close_ready)
    assert window.coordinator.cancelled
    assert window.worker_stopped.is_set()
    assert window.process.state() == QProcess.ProcessState.NotRunning


def test_persian_answer_uses_rtl_text_direction(application, window):
    window.prompt.setPlainText("persian")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    assert "فارسی" in window.answer.toPlainText()
    assert window.answer.document().defaultTextOption().textDirection() == Qt.LayoutDirection.RightToLeft


def test_update_preserves_incremental_preparation(application, window):
    window.update_library()
    assert window.coordinator.running
    assert not window.ask.isEnabled()
    window.coordinator.running = False
    window._preparation_progress()
    wait_for(application, lambda: window.checker is None)
    assert window.ready and window.ask.isEnabled()


def test_worker_rejects_missing_library_without_loading_models(tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing library must not load a model")

    with pytest.raises(FileNotFoundError):
        run_request(
            {"database": str(tmp_path / "missing.sqlite"), "prompt": "question"},
            lambda event: None,
            embedder_factory=forbidden,
        )


def test_worker_keeps_citation_errors_out_of_final_events(tmp_path, monkeypatch):
    database = tmp_path / "present.sqlite"
    database.touch()

    def invalid(*args, **kwargs):
        yield "Tentative answer"
        raise RuntimeError("The generated answer has no source citations")

    monkeypatch.setattr("studyrag.desktop.worker.stream_answer_question", invalid)
    events = []
    with pytest.raises(RuntimeError, match="no source citations"):
        run_request(
            {"database": str(database), "prompt": "question"},
            events.append,
            embedder_factory=lambda *a, **kw: None,
            generator_factory=lambda *a: None,
        )
    assert [event["type"] for event in events] == ["status", "model", "token"]


def test_worker_sends_validated_answer_and_real_source_metadata(tmp_path, monkeypatch):
    database = tmp_path / "present.sqlite"
    database.touch()
    source = tmp_path / "lesson.pdf"

    def valid(*args, **kwargs):
        yield "An explanation "
        yield StudyAnswer("An explanation [S1]", (AnswerSource("S1", source, 7, 1.0),))

    monkeypatch.setattr("studyrag.desktop.worker.stream_answer_question", valid)
    events = []
    run_request(
        {"database": str(database), "prompt": "question"},
        events.append,
        embedder_factory=lambda *a, **kw: None,
        generator_factory=lambda *a: None,
    )
    assert [event["type"] for event in events] == ["status", "model", "token", "done"]
    assert events[-1]["sources"] == [
        {"id": "S1", "file": "lesson.pdf", "path": str(source), "page": 7},
    ]


def test_followup_keeps_transcript_sends_history_and_scopes_sources(application, window):
    window.prompt.setPlainText("Explain sampling bias")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    window.prompt.setPlainText("Give me an example")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    transcript = window.answer.toPlainText()
    assert "Explain sampling bias" in transcript and "Give me an example" in transcript
    assert transcript.count("Final answer [S1]") == 2
    assert window.sources.topLevelItemCount() == 2
    assert window.sources.topLevelItem(0).text(0) == "Answer 1"
    assert window.sources.topLevelItem(1).text(0) == "Answer 2"
    requests = Path(window.worker_program[1][-1]).with_suffix(".requests.jsonl")
    payloads = [json.loads(line) for line in requests.read_text(encoding="utf-8").splitlines()]
    assert payloads[0]["history"] == []
    assert payloads[1]["history"][0]["question"] == "Explain sampling bias"
    assert len(window.chat_store.turns(window.chat_id)) == 2
    assert window.prompt.toPlainText() == ""


def test_new_chat_is_empty_and_previous_chat_can_be_resumed(application, window):
    window.prompt.setPlainText("First topic")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    previous = window.chat_id
    window.new_chat()
    assert window.chat_id != previous
    assert not window.turns and window.answer.toPlainText() == ""
    window.chat_selector.setCurrentIndex(window.chat_selector.findData(previous))
    assert window.chat_id == previous
    assert "First topic" in window.answer.toPlainText()


def test_failed_followup_preserves_completed_history(application, window):
    window.prompt.setPlainText("First topic")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    window.prompt.setPlainText("invalid citation")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    assert "First topic" in window.answer.toPlainText()
    assert "Final answer [S1]" in window.answer.toPlainText()
    assert "Partial answer" not in window.answer.toPlainText()
    assert "invalid citation" not in window.answer.toPlainText()
    assert len(window.chat_store.turns(window.chat_id)) == 1
    assert window.prompt.toPlainText() == "invalid citation"


def test_saved_chat_restores_after_window_restart(application, window, tmp_path):
    window.prompt.setPlainText("Remember this topic")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    restored = desktop.StudyWindow(
        database=window.database,
        material=Path(window.folder.text()),
        coordinator=Coordinator(tmp_path),
        worker_program=window.worker_program,
    )
    try:
        restored.show()
        wait_for(application, lambda: restored.checker is None)
        assert "Remember this topic" in restored.answer.toPlainText()
        assert restored.sources.topLevelItem(0).child(0).text(0) == "S1"
    finally:
        restored.close()
        wait_for(application, lambda: restored.close_ready)
        restored.deleteLater()


def test_worker_rejects_malformed_history_before_model_load(tmp_path):
    database = tmp_path / "index.sqlite"
    database.touch()

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid history must not load models")

    with pytest.raises(ValueError, match="history"):
        run_request(
            {"database": str(database), "prompt": "question", "history": "invalid"},
            lambda event: None,
            embedder_factory=forbidden,
        )


def test_done_event_followed_by_failed_process_does_not_save_turn(application, window):
    window.prompt.setPlainText("badexit")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    assert window.chat_store.turns(window.chat_id) == ()
    assert "Final answer" not in window.answer.toPlainText()


def test_mixed_language_chat_keeps_direction_for_each_answer(application, window):
    window.prompt.setPlainText("English question")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    window.prompt.setPlainText("persian")
    window.ask_question()
    wait_for(application, lambda: not window._busy())
    block = window.answer.document().begin()
    checked = 0
    while block.isValid():
        if "Final answer" in block.text():
            assert block.blockFormat().layoutDirection() == Qt.LayoutDirection.LeftToRight
            checked += 1
        if "فارسی" in block.text():
            assert block.blockFormat().layoutDirection() == Qt.LayoutDirection.RightToLeft
            checked += 1
        block = block.next()
    assert checked == 2


def test_transcript_markdown_cannot_load_local_images(window, tmp_path):
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QImage, QTextDocument

    path = tmp_path / "private-image.png"
    image = QImage(10, 10, QImage.Format.Format_RGB32)
    image.fill(Qt.GlobalColor.red)
    assert image.save(str(path))
    url = QUrl.fromLocalFile(str(path))
    window._insert_message("StudyRag", f"![private image]({url.toString()})", markdown=True)
    assert "[Image omitted]" in window.answer.toPlainText()
    assert "\ufffc" not in window.answer.toPlainText()
    resource = window.answer.document().resource(QTextDocument.ResourceType.ImageResource, url)
    assert resource is None or resource.isNull() or resource.isEmpty()
