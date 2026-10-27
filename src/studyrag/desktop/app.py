import codecs
from contextlib import closing
import json
from pathlib import Path
import re
import sqlite3
import sys
from threading import Event

from PySide6.QtCore import QByteArray, QProcess, QProcessEnvironment, QThread, QTimer, Qt, QUrl, Signal
from PySide6.QtGui import (
    QDesktopServices,
    QFont,
    QTextCursor,
    QTextDocument,
    QTextDocumentFragment,
    QTextBlockFormat,
    QTextCharFormat,
)
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from studyrag.preparation import PreparationCoordinator
from studyrag.settings import (
    DEFAULT_DATABASE,
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MATERIAL_FOLDER,
    PROJECT_ROOT,
    PACKAGE_IMPORT_ROOT,
    worker_python,
)
from studyrag.study import OllamaGenerator, StudyMode
from studyrag.study.conversations import ConversationStore, SavedTurn, bounded_history


class LibraryCheck(QThread):
    result = Signal(bool, str, str)

    def __init__(self, coordinator, parent=None):
        super().__init__(parent)
        self.coordinator = coordinator

    def run(self):
        try:
            ready = self.coordinator._is_fresh()
            counts = "No prepared library"
            database = self.coordinator.database_path
            if database.exists():
                with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
                    values = [
                        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        for table in ("documents", "pages", "chunks")
                    ]
                counts = f"{values[0]} documents · {values[1]} pages · {values[2]} chunks"
            message = "Library ready" if ready else "Update the library before asking a question."
            self.result.emit(ready, message, counts)
        except Exception as error:
            self.result.emit(False, str(error), "Library unavailable")


class Cleanup(QThread):
    warning = Signal(str)

    def __init__(self, coordinator, unload_model, worker_stopped, parent=None):
        super().__init__(parent)
        self.coordinator = coordinator
        self.unload_model = unload_model
        self.worker_stopped = worker_stopped

    def run(self):
        self.coordinator.cancel()
        preparation = self.coordinator._thread
        if preparation is not None:
            preparation.join()
        self.worker_stopped.wait()
        if self.unload_model:
            try:
                OllamaGenerator(DEFAULT_GENERATION_MODEL).unload()
            except Exception as error:
                self.warning.emit(f"Model unload warning: {error}")


class TranscriptDocument(QTextDocument):
    def loadResource(self, resource_type, url):
        # Block resource access from both Markdown fragments and the transcript.
        # Cache a valid empty value: None permits Qt fallback resource loading.
        empty = QByteArray()
        self.addResource(resource_type, url, empty)
        return empty


class AnswerView(QTextBrowser):
    def __init__(self):
        super().__init__()
        self._safe_document = TranscriptDocument(self)
        self.setDocument(self._safe_document)

    def loadResource(self, resource_type, url):
        # Model output cannot load remote images or arbitrary local image files.
        return None


class StudyWindow(QMainWindow):
    def __init__(
        self,
        *,
        database=DEFAULT_DATABASE,
        material=DEFAULT_MATERIAL_FOLDER,
        project_root=PROJECT_ROOT,
        coordinator=None,
        worker_program=None,
        chat_database=None,
    ):
        super().__init__()
        self.project_root = Path(project_root).resolve()
        self.database = Path(database).resolve()
        self.chat_store = ConversationStore(
            Path(chat_database) if chat_database is not None else self.database.with_name("chats.sqlite")
        )
        self.chat_id = None
        self.chat_scope = ""
        self.turns = ()
        self.current_question = ""
        self.current_mode = "answer"
        self.pending_answer = None
        self.coordinator = coordinator or PreparationCoordinator(
            self.project_root,
            Path(material),
            self.database,
        )
        self.worker_program = worker_program or (
            worker_python(),
            ["-B", "-m", "studyrag.desktop.worker"],
        )
        self.ready = False
        self.checker = None
        self.cleanup = None
        self.closing = False
        self.close_ready = False
        self.model_used = False
        self.worker_stopped = Event()
        self.worker_stopped.set()
        self.draft = ""
        self.completed = False
        self.cancelled = False
        self.failure = ""
        self.pending = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.setWindowTitle("StudyRag — Local study assistant")
        self.resize(1000, 790)
        self.setMinimumSize(760, 620)
        self._build_ui(Path(material))
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_output)
        self.process.readyReadStandardError.connect(self._read_error)
        self.process.finished.connect(self._study_finished)
        self.process.errorOccurred.connect(self._process_error)
        self.preparation_timer = QTimer(self)
        self.preparation_timer.setInterval(150)
        self.preparation_timer.timeout.connect(self._preparation_progress)
        self.shutdown_timer = QTimer(self)
        self.shutdown_timer.setInterval(100)
        self.shutdown_timer.timeout.connect(self._finish_close)
        self._load_chats()
        self._controls()
        QTimer.singleShot(0, self.check_library)

    def _build_ui(self, material):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(18, 16, 18, 16)
        title = QLabel("StudyRag")
        title.setFont(QFont("Segoe UI", 19, QFont.Weight.DemiBold))
        layout.addWidget(title)
        layout.addWidget(QLabel("Study your PDFs with local models and page references."))

        library = QGroupBox("Library")
        form = QVBoxLayout(library)
        row = QHBoxLayout()
        self.folder = QLineEdit(str(material))
        self.folder.setAccessibleName("Learning material folder")
        self.folder.textEdited.connect(self._folder_edited)
        self.browse = QPushButton("Choose folder…")
        self.browse.clicked.connect(self.choose_folder)
        self.update_button = QPushButton("Update library")
        self.update_button.clicked.connect(self.update_library)
        row.addWidget(self.folder, 1)
        row.addWidget(self.browse)
        row.addWidget(self.update_button)
        form.addLayout(row)
        self.library_label = QLabel("Checking library…")
        self.library_label.setWordWrap(True)
        self.counts_label = QLabel("")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        form.addWidget(self.library_label)
        form.addWidget(self.counts_label)
        form.addWidget(self.progress)
        layout.addWidget(library)

        study = QGroupBox("Study chat")
        study_layout = QVBoxLayout(study)
        chat_row = QHBoxLayout()
        chat_row.addWidget(QLabel("Conversation"))
        self.chat_selector = QComboBox()
        self.chat_selector.setAccessibleName("Saved conversations")
        self.chat_selector.setMinimumContentsLength(20)
        self.chat_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.chat_selector.currentIndexChanged.connect(self._select_chat)
        chat_row.addWidget(self.chat_selector, 1)
        self.new_chat_button = QPushButton("New chat")
        self.new_chat_button.clicked.connect(self.new_chat)
        chat_row.addWidget(self.new_chat_button)
        study_layout.addLayout(chat_row)
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode"))
        self.mode = QComboBox()
        self.mode.setAccessibleName("Study mode")
        self.mode.addItems([mode.value.capitalize() for mode in StudyMode])
        mode_row.addWidget(self.mode)
        mode_row.addStretch()
        mode_row.addWidget(QLabel(f"Ollama: {DEFAULT_GENERATION_MODEL}"))
        study_layout.addLayout(mode_row)
        self.prompt = QPlainTextEdit()
        self.prompt.setAccessibleName("Question or topic")
        self.prompt.setPlaceholderText("Ask a question or continue the conversation: ‘Give me an example.’")
        self.prompt.setMaximumHeight(100)
        actions = QHBoxLayout()
        self.ask = QPushButton("Send")
        self.ask.clicked.connect(self.ask_question)
        self.cancel_button = QPushButton("Cancel answer")
        self.cancel_button.clicked.connect(self.cancel_answer)
        actions.addWidget(self.ask)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        self.answer_title = QLabel("Conversation")
        study_layout.addWidget(self.answer_title)
        splitter = QSplitter(Qt.Orientation.Vertical)
        self.answer = AnswerView()
        self.answer.setAccessibleName("Study conversation")
        self.answer.setOpenExternalLinks(False)
        self.answer.setOpenLinks(False)
        self.sources = QTreeWidget()
        self.sources.setAccessibleName("PDF page references")
        self.sources.setHeaderLabels(["Answer / source", "PDF", "Page"])
        self.sources.setRootIsDecorated(True)
        self.sources.itemDoubleClicked.connect(self.open_source)
        splitter.addWidget(self.answer)
        splitter.addWidget(self.sources)
        splitter.setSizes([270, 90])
        study_layout.addWidget(splitter, 1)
        study_layout.addWidget(self.prompt)
        study_layout.addLayout(actions)
        layout.addWidget(study, 1)
        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)
        self.setCentralWidget(central)

    def _busy(self):
        return self.process.state() != QProcess.ProcessState.NotRunning

    def _scope(self):
        return str(self.database) + "\n" + str(Path(self.folder.text()).expanduser().resolve())

    def _load_chats(self, selected=None):
        self.chat_scope = self._scope()
        chats = self.chat_store.list(self.chat_scope)
        if not chats:
            selected = self.chat_store.create(self.chat_scope)
            chats = self.chat_store.list(self.chat_scope)
        selected = selected or chats[0][0]
        self.chat_selector.blockSignals(True)
        self.chat_selector.clear()
        for identifier, title in chats:
            self.chat_selector.addItem(title, identifier)
        index = self.chat_selector.findData(selected)
        self.chat_selector.setCurrentIndex(max(index, 0))
        self.chat_selector.blockSignals(False)
        self._select_chat(self.chat_selector.currentIndex())

    def _select_chat(self, index):
        if self._busy():
            return
        identifier = self.chat_selector.itemData(index)
        if identifier is None:
            return
        self.chat_id = identifier
        self.turns = self.chat_store.turns(identifier)
        self.prompt.clear()
        self._render_conversation()
        self.answer_title.setText("Conversation")

    def new_chat(self):
        if self._busy() or self.closing:
            return
        identifier = self.chat_store.create(self._scope())
        self._load_chats(identifier)
        self.message.setText("New conversation. Earlier chats are saved locally.")

    def _insert_message(self, heading, text, *, markdown=False):
        cursor = QTextCursor(self.answer.document())
        cursor.movePosition(QTextCursor.MoveOperation.End)
        if not self.answer.document().isEmpty():
            cursor.insertBlock()
        header = QTextCharFormat()
        header.setFontWeight(QFont.Weight.DemiBold)
        cursor.insertText(heading, header)
        plain = QTextCharFormat()
        plain.setFontWeight(QFont.Weight.Normal)
        cursor.insertBlock(QTextBlockFormat(), plain)
        start = cursor.position()
        if markdown:
            document = TranscriptDocument()
            document.setDefaultFont(self.answer.font())
            document.setMarkdown(text, QTextDocument.MarkdownFeature.MarkdownNoHTML)
            # Qt's image layout can fall back to loading a file directly even
            # when a resource callback returns empty data. Keep the transcript
            # text-only by removing embedded objects before copying the fragment.
            while True:
                image = document.find("\ufffc")
                if image.isNull():
                    break
                image.insertText("[Image omitted]", QTextCharFormat())
            cursor.insertFragment(QTextDocumentFragment(document))
        else:
            cursor.insertText(text)
        end = cursor.position()
        cursor.setPosition(start)
        cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
        block = QTextBlockFormat()
        block.setLayoutDirection(
            Qt.LayoutDirection.RightToLeft
            if re.search(r"[\u0600-\u06ff]", text)
            else Qt.LayoutDirection.LeftToRight
        )
        cursor.mergeBlockFormat(block)
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertBlock(QTextBlockFormat(), plain)

    def _add_sources(self, number, question, sources):
        if not sources:
            return
        group = QTreeWidgetItem([f"Answer {number}", question[:80], ""])
        self.sources.addTopLevelItem(group)
        for source in sources:
            item = QTreeWidgetItem([source["id"], source["file"], str(source["page"])])
            item.setData(0, Qt.ItemDataRole.UserRole, source["path"])
            item.setToolTip(1, source["path"])
            group.addChild(item)
        group.setExpanded(True)

    def _render_conversation(self, *, pending=False):
        self.answer.clear()
        self.sources.clear()
        for number, turn in enumerate(self.turns, 1):
            self._insert_message(f"You · {number}", turn.question)
            self._insert_message(f"StudyRag · {number}", turn.answer, markdown=True)
            self._add_sources(number, turn.question, turn.sources)
        if pending:
            number = len(self.turns) + 1
            self._insert_message(f"You · {number}", self.current_question)
            text = self.pending_answer["answer"] if self.pending_answer else ""
            self._insert_message(f"StudyRag · {number}", text, markdown=bool(text))
            if self.pending_answer:
                self._add_sources(number, self.current_question, self.pending_answer["sources"])
        latest = (
            self.pending_answer["answer"]
            if pending and self.pending_answer
            else (self.turns[-1].answer if self.turns else "")
        )
        self._text_direction(latest)
        self.answer.moveCursor(QTextCursor.MoveOperation.End)
        self.sources.resizeColumnToContents(0)

    def _controls(self):
        busy = self._busy()
        preparing = bool(self.coordinator.status()["running"])
        checking = self.checker is not None
        idle = not (busy or preparing or checking or self.closing)
        for widget in (self.folder, self.browse, self.update_button):
            widget.setEnabled(idle)
        for widget in (self.prompt, self.mode, self.ask):
            widget.setEnabled(idle and self.ready)
        self.cancel_button.setEnabled(busy and not self.closing)
        self.chat_selector.setEnabled(idle)
        self.new_chat_button.setEnabled(idle)

    def check_library(self):
        if self.closing or self.checker is not None:
            return
        self.ready = False
        self.checker = LibraryCheck(self.coordinator, self)
        self.checker.result.connect(self._library_checked)
        self.checker.finished.connect(self._check_finished)
        self.checker.start()
        self._controls()

    def _check_finished(self):
        checker = self.checker
        self.checker = None
        checker.deleteLater()
        self._controls()

    def _library_checked(self, ready, message, counts):
        if self.closing:
            return
        self.ready = ready
        self.library_label.setText(message)
        self.counts_label.setText(counts)
        self.progress.setValue(100 if ready else 0)

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(self, "Choose PDF folder", self.folder.text())
        if selected:
            self.folder.setText(selected)
            self.coordinator.material_folder = Path(selected).resolve()
            self._load_chats()
            self.check_library()

    def _folder_edited(self):
        self.ready = False
        self.library_label.setText("Click Update library to use this folder.")
        self._controls()

    def update_library(self):
        folder = Path(self.folder.text()).expanduser().resolve()
        if not folder.is_dir():
            self.message.setText("Choose an existing folder containing your PDFs.")
            return
        if self.chat_scope != self._scope():
            self._load_chats()
        if self.coordinator.start(folder, force=False):
            self.ready = False
            self.library_label.setText("Updating changed documents; reusing completed work…")
            self.preparation_timer.start()
            self._controls()

    def _preparation_progress(self):
        state = self.coordinator.status()
        self.progress.setValue(int(state["percent"]))
        self.library_label.setText(str(state["message"]))
        if not state["running"]:
            self.preparation_timer.stop()
            if state["error"]:
                self.message.setText(str(state["error"]))
            self.check_library()

    def ask_question(self):
        if not self.ready or self._busy() or self.closing:
            return
        question = self.prompt.toPlainText().strip()
        if not question:
            self.message.setText("Enter a question or topic first.")
            return
        self.completed = self.cancelled = False
        self.failure = self.pending = self.draft = ""
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.current_question = question
        self.current_mode = self.mode.currentText().lower()
        self.pending_answer = None
        self._render_conversation(pending=True)
        self.answer_title.setText("Generating response — citations pending")
        self.message.setText("Starting the local study worker…")
        environment = QProcessEnvironment.systemEnvironment()
        for key, value in {
            "PYTHONPATH": str(PACKAGE_IMPORT_ROOT),
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
        }.items():
            environment.insert(key, value)
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(self.project_root))
        self.worker_stopped.clear()
        self.process.start(self.worker_program[0], self.worker_program[1])
        request = {
            "database": str(self.database),
            "prompt": question,
            "mode": self.current_mode,
            "history": bounded_history(
                [
                    {"question": turn.question, "answer": turn.answer, "topic": turn.topic}
                    for turn in self.turns[-4:]
                ]
            ),
        }
        self.process.write((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"))
        self.process.closeWriteChannel()
        self._controls()

    def _read_output(self):
        try:
            self.pending += self.decoder.decode(bytes(self.process.readAllStandardOutput()))
            while "\n" in self.pending:
                line, self.pending = self.pending.split("\n", 1)
                if line.strip():
                    self._event(json.loads(line))
        except (ValueError, UnicodeError, KeyError, TypeError) as error:
            self.failure = f"Invalid study worker output: {error}"
            self.process.kill()

    def _read_error(self):
        details = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if details.strip():
            print(details, file=sys.stderr, end="", flush=True)

    def _event(self, event):
        if self.cancelled or self.closing:
            return
        if self.completed:
            raise ValueError("Received output after the final answer")
        kind = event["type"]
        if kind in {"status", "model"}:
            self.model_used |= kind == "model"
            self.message.setText(event["message"])
        elif kind == "token":
            self.draft += event["text"]
            self.answer.moveCursor(QTextCursor.MoveOperation.End)
            self.answer.insertPlainText(event["text"])
            self._text_direction(self.draft)
        elif kind == "done":
            self.completed = True
            self.pending_answer = event
            self._render_conversation(pending=True)
            self.sources.resizeColumnToContents(1)
            self.answer_title.setText("Answer — citations checked")
            self.message.setText("Double-click a source to open its PDF.")
        elif kind == "error":
            self.failure = event["message"]
        else:
            raise ValueError("Unknown study worker event")

    def _text_direction(self, text):
        direction = (
            Qt.LayoutDirection.RightToLeft
            if re.search(r"[\u0600-\u06ff]", text)
            else Qt.LayoutDirection.LeftToRight
        )
        option = self.answer.document().defaultTextOption()
        option.setTextDirection(direction)
        self.answer.document().setDefaultTextOption(option)

    def _study_finished(self, exit_code, exit_status):
        self._read_output()
        try:
            self.pending += self.decoder.decode(b"", final=True)
        except UnicodeError:
            self.failure = "The study worker ended with incomplete text output."
        if self.pending.strip() and not self.failure:
            try:
                self._event(json.loads(self.pending))
            except (ValueError, KeyError, TypeError) as error:
                self.failure = str(error)
        self.pending = ""
        self.worker_stopped.set()
        if self.cancelled or self.failure or not self.completed or exit_code:
            self.pending_answer = None
            self._render_conversation()
            self.answer_title.setText("Conversation")
            self.message.setText(
                "Answer cancelled."
                if self.cancelled
                else self.failure or "The study worker stopped before completing the answer."
            )
        else:
            try:
                self.chat_store.append(
                    self.chat_id,
                    SavedTurn(
                        self.current_question,
                        self.pending_answer["answer"],
                        self.current_mode,
                        tuple(self.pending_answer["sources"]),
                        self.pending_answer.get("topic", self.current_question),
                    ),
                )
                self.pending_answer = None
                self._load_chats(self.chat_id)
                self.answer_title.setText("Conversation — citations checked")
            except (sqlite3.Error, ValueError, OSError) as error:
                self.message.setText(f"Answer received, but the conversation could not be saved: {error}")
        self._controls()

    def _process_error(self, error):
        if error == QProcess.ProcessError.FailedToStart:
            self.worker_stopped.set()
            self.failure = "Could not start the study worker: " + self.process.errorString()
            self.message.setText(self.failure)
            self._render_conversation()
            self._controls()

    def cancel_answer(self):
        if self._busy():
            self.cancelled = True
            self.process.terminate()
            QTimer.singleShot(1500, self._kill_worker)

    def _kill_worker(self):
        if self._busy() and (self.cancelled or self.closing):
            self.process.kill()

    def open_source(self, item, column=0):
        stored_path = item.data(0, Qt.ItemDataRole.UserRole)
        if not stored_path:
            return
        path = Path(stored_path)
        if not path.is_file() or not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.message.setText("The source PDF could not be opened.")

    def closeEvent(self, event):
        if self.close_ready:
            event.accept()
            return
        event.ignore()
        if self.closing:
            return
        self.closing = True
        self.preparation_timer.stop()
        self.message.setText("Closing StudyRag and stopping its workers…")
        self.cancel_answer()
        self._controls()
        self.cleanup = Cleanup(self.coordinator, self.model_used, self.worker_stopped, self)
        self.cleanup.warning.connect(lambda message: print(message, file=sys.stderr))
        self.cleanup.start()
        self.shutdown_timer.start()

    def _finish_close(self):
        if self._busy() or self.checker is not None or self.cleanup.isRunning():
            return
        self.shutdown_timer.stop()
        self.close_ready = True
        self.close()


def launch(*, database=DEFAULT_DATABASE, material=DEFAULT_MATERIAL_FOLDER) -> int:
    application = QApplication.instance() or QApplication(sys.argv[:1])
    application.setApplicationName("StudyRag")
    application.setFont(QFont("Segoe UI", 10))
    window = StudyWindow(database=database, material=material)
    window.show()
    return application.exec()
