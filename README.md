# StudyRag

StudyRag is a local desktop app for studying English, German and Persian PDFs.
It extracts text, tables and formulas, searches your library, and answers questions
through Ollama with PDF and page references. You can continue a conversation,
ask for an explanation, compare topics or generate a quiz.

Version 0.1.0 is an early release for Windows and Python 3.12.

## Setup

You need 64-bit Python 3.12, a compatible NVIDIA GPU runtime and a running
[Ollama](https://ollama.com/) installation. The current OCR backend requires
PaddlePaddle GPU; the window can open without loading it.

From the project directory, on the Windows CUDA 12.6 setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
New-Item -ItemType Directory -Force material
ollama pull qwen3.5:9b
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"
```

The last two commands download the language and embedding models. OCR models
are downloaded to `models/paddlex/` on first use, unless
`PADDLE_PDX_CACHE_HOME` points elsewhere. A compatible NVIDIA driver and the
runtime libraries required by PaddlePaddle must already be installed. The full
AI/GPU stack has not been validated with a fresh install.

## Using the app

1. Double-click `StudyRag.cmd`.
2. Select a folder containing your PDFs and click **Update library**.
3. Choose a study mode, type a question and press **Send**.
4. Continue with follow-ups, or use **New chat** to start another conversation.

The conversation selector reopens saved chats. Each answer has its own sources;
double-click a source to open its PDF. The cited page is shown in the source list.
Persian messages use right-to-left text direction.

This is a native PySide6 window. It does not need a browser or an application web
server. OCR, indexing and generation run in child processes so the window stays
responsive. Closing it stops its workers. After study requests, the app attempts
to unload the configured generation model without stopping the Ollama service.
**Updating the library unloads all loaded Ollama models before OCR** to free GPU
memory, which can affect other applications using that service.

Completed answers and their sources are saved in `data/chats.sqlite`. Cancelled
or failed drafts are not saved. Chats belong to the selected library and survive
index updates. A follow-up retrieves fresh passages and includes up to four
recent completed turns, within a 10,000-character history budget. Older messages
remain saved but may fall outside the model's context.

For a console with diagnostics:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -B -m studyrag desktop
```

Launcher errors and worker diagnostics go to `data/desktop.log`.
`StudyRag.cmd --check` checks Python and Qt without starting the app or models.

## Extraction and retrieval

PyMuPDF reads pages with a usable text layer. Sparse text, substantial images or
vector graphics, and detected mathematical content trigger full-page
PP-StructureV3 OCR. Native text is retained alongside the OCR result.

The OCR pipeline uses mobile text detection and Latin/Arabic recognizers,
PP-FormulaNet_plus-L for formulas, and separate table structure and cell models.
It keeps formula LaTeX and table HTML in the index. Text lines are recognized in
both scripts, with confidence used to select a result. Chart interpretation is
disabled. Routing and script selection use heuristics and can make mistakes.

Images are limited to a 2,000-pixel longest side, text detection to 1,536 pixels,
and inference batches to one. Paddle defaults to a 6,144 MB allocation budget;
CUDA and driver allocations sit outside that budget. GPU errors are reported
rather than retried on CPU.

Unchanged PDFs reuse extracted pages, blocks and chunks when their hashes,
paths and processing settings match. Unchanged chunks reuse embeddings for the
same model. Interrupted updates keep completed documents for the next attempt;
the unfinished document starts again. Removing a PDF from the selected folder
removes its index entries without deleting the source file.

Retrieval combines `intfloat/multilingual-e5-small` embeddings with SQLite FTS5
keyword search. Answers use `qwen3.5:9b` by default. All models are pretrained;
indexing your PDFs does not fine-tune them.

An update builds `data/studyrag.building.sqlite`, validates it, and activates it
as `data/studyrag.sqlite`. The preceding library is kept as
`data/studyrag.previous.sqlite`. The active library stays available while its
replacement is being built. Changes to the extraction policy, schema or OCR
model require another **Update library** before new results take effect.

See [SPEC.md](SPEC.md) for the design constraints and
`src/studyrag/settings.py` for defaults. Source checkouts store local data beside
the source. Installed wheels use `%LOCALAPPDATA%\StudyRag` on Windows or XDG user
data storage on Unix. Set `STUDYRAG_HOME` to choose another writable location.

## Quality limits

Check important numbers and formulas against the PDF. Persian digits, word
boundaries and formula symbols can be misread, and full-page OCR can be less
accurate than a page's native text. Earlier spot checks used the Plus S formula
recognizer; the current Plus L recognizer has not been benchmarked here.

Answers stream as tentative text. The final answer is accepted only if its
citation markers refer to retrieved sources. This validates marker IDs, not the
truth of every claim. Follow-up topic detection is also heuristic and can miss
ambiguous references.

Offline tests cover the backend, streaming and Qt interactions using simulated
models. They do not measure OCR speed, VRAM use or live multi-turn Qwen quality.

## Command line

These commands use the same library defaults as the desktop app:

```powershell
.\.venv\Scripts\python.exe -m studyrag scan .\material
.\.venv\Scripts\python.exe -m studyrag ingest .\material
.\.venv\Scripts\python.exe -m studyrag index
.\.venv\Scripts\python.exe -m studyrag search "What is correlation?" --top-k 5
.\.venv\Scripts\python.exe -m studyrag ask "Explain correlation" --mode explain
```

The CLI buffers the final answer; the desktop streams it. `desktop` accepts
`--database` and `--material` for alternate paths. Without an editable install,
set `PYTHONPATH=src` before running the commands.

## Development

The offline suite can run without Torch, Paddle or downloaded models:

```powershell
py -3.12 -m venv .test-venv
.\.test-venv\Scripts\python.exe -m pip install -r requirements-test.txt
.\.test-venv\Scripts\python.exe -m pip install --no-deps -e .
$env:QT_QPA_PLATFORM = "offscreen"
New-Item -ItemType Directory -Force work
.\.test-venv\Scripts\python.exe -m ruff check src scripts
.\.test-venv\Scripts\python.exe -m ruff format --check src scripts tests
.\.test-venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp work/pytest
.\.test-venv\Scripts\python.exe -m build
.\.test-venv\Scripts\python.exe scripts/check_release.py
```

GitHub Actions runs the offline checks on Windows and Ubuntu with Python 3.12.

The Python package is split into `desktop/`, `ingestion/`, `retrieval/` and
`study/`. `preparation.py` coordinates library updates. Tests live in `tests/`.

To evaluate retrieval on your own material, copy
`docs/retrieval-cases.example.json` to `work/`, replace its queries and expected
filename fragments, then run:

```powershell
.\.venv\Scripts\python.exe scripts/benchmark_retrieval.py --database data/studyrag.sqlite --cases work/my-cases.json
```

This measures source retrieval, not generated answer correctness.

## Source releases

Run `python scripts/check_release.py` before staging files. It audits the public
source set, checks for tracked private files, and looks for common credential
formats without printing their values. The credential check is heuristic.

To export a new source archive:

```powershell
.\.venv\Scripts\python.exe scripts/check_release.py --archive dist/StudyRag-source-v0.1.0.zip
```

The command refuses to overwrite an existing archive. PDFs, databases, chats,
logs, model weights, environments, backups and working files are excluded from
Git and source exports. Avoid uploading an unfiltered copy of the local folder.

No project license has been assigned. Dependency and model terms still apply;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
