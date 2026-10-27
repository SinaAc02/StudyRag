# StudyRag design

StudyRag is a desktop app for studying English, German and Persian learning and
reference PDFs. PySide6 Qt Widgets provides the interface; Python services handle
document preparation, retrieval and generation. The command-line tools expose
the same services for scanning, ingestion, indexing, search and questions.

## Document preparation

The app discovers PDFs in the selected folder and hashes their contents. PyMuPDF
extracts native text and positioned blocks. Pages with sparse or damaged text,
substantial visual content or detected mathematics go through full-page
PP-StructureV3 OCR. The index stores the native extraction alongside OCR results.

OCR uses mobile Latin and Arabic/Persian text recognizers, dedicated table
models and PP-FormulaNet_plus-L. Table HTML and formula LaTeX are stored with
page blocks. Image dimensions, batch size and Paddle allocations are bounded to
fit the target 12 GB GPU. GPU inference failures stop preparation and report an
error.

Chunks belong to a single document and page. Logical region IDs associate
related blocks for retrieval. SQLite stores documents, pages, blocks, chunks,
processing keys and embedding vectors.

Matching file hashes, paths and processing keys allow complete documents to be
reused. Vectors are reused for unchanged chunks and the same embedding model.
An interrupted build resumes from its completed documents. Removed sources are
pruned from the index; the selected PDF folder remains under the user's control.

Preparation writes a staging database. Activation checks source hashes,
extraction metadata, schema and processing versions, and embeddings. A validated
build becomes the active library, with the preceding library kept as a backup.
SQLite's online backup API coordinates activation with readers on Windows.

## Search and answers

Retrieval combines normalized multilingual E5 embeddings with SQLite FTS5 ranks.
E5 inputs use the model's query and passage prefixes. Matching logical regions
are deduplicated and expanded before their passages are supplied to generation.

Answer, explain, compare and quiz modes use a local Ollama model. Output streams
into the transcript as a draft. Citation checks resolve source markers to the
retrieved PDFs and page numbers. A successful answer is rendered as Markdown
with its sources; incomplete streams and citation errors discard the draft and
display an error. Persian messages use right-to-left text direction.

Follow-ups use Ollama chat messages and retrieve fresh passages for each
question. A topic heuristic connects common short follow-ups to the recent
conversation. Up to four completed turns fit within a 10,000-character history
budget, alongside bounded fresh passages in a 16,384-token context window.
Each answer's source IDs refer to that turn's retrieved passages.

## Desktop and storage

The window provides a folder picker, library status and progress, study modes,
a chat transcript, saved-chat selection, a message composer, cancellation and
sources grouped by answer. Startup checks library readiness. **Update library**
starts preparation.

Completed turns and their sources are saved after successful worker completion
in a separate `chats.sqlite` database. Chats are scoped to the document index
and material folder and survive index replacement. Failed or cancelled drafts
are discarded while earlier messages remain available.

OCR, indexing and generation run in child processes. Study requests travel over
stdin as JSON; workers return JSON events over stdout. Closing the window stops
its workers. Completion or cancellation releases the embedding process, and
shutdown attempts to unload the configured generation model. Preparation
unloads loaded Ollama models before OCR to free GPU memory.

## Testing

The automated suite covers document reuse, activation failures, OCR routing,
streaming, citations, chat persistence and Qt worker lifecycle. Model fixtures
exercise these paths offline. OCR accuracy, GPU performance and live answer
quality require evaluation on representative PDFs with the installed models.
