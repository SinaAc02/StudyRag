# StudyRag specification

## Goal and scope

A simple native desktop assistant for studying English, German and Persian
learning/source PDFs. The UI is PySide6 Qt Widgets; application services remain
Python modules. No browser, web server or heartbeat is part of the desktop path.
Example exams are not a supported separate input or a generation target.

## Extraction and persistence

- Discover/hash PDFs without modifying source files.
- Preserve native text, page references and positioned text blocks.
- Use native extraction for usable text and selective full-page PP-StructureV3 for
  sparse/damaged text or substantive visual/math content.
- Preserve formula LaTeX and table HTML with dedicated recognition models; use
  PP-FormulaNet_plus-L for formulas, mobile Latin/Arabic OCR, bounded images,
  batch size one and a Paddle GPU
  allocation budget to leave headroom on the 12 GB GPU. Do not retry on CPU.
- Keep native evidence even when OCR supplies searchable text.
- Keep chunks within document/page boundaries and record logical regions.
- Persist pages, blocks, chunks, processing keys and vectors in SQLite.
- Cache complete unchanged documents and vectors for unchanged chunks/models.
- Invalidate incompatible extraction settings; prune sources absent from the folder.
- Resume completed documents from compatible interrupted staging builds.
- Retain the active library until a validated replacement is ready; keep a backup.
- Recheck source hashes and schema/policy version before activation.

## Retrieval and study

- Use multilingual E5 query/passage prefixes and normalized vectors.
- Combine semantic retrieval and FTS5 ranks; deduplicate/expand logical regions.
- Retrieve bounded evidence before answer, explain, compare and quiz generation.
- Use an installed local Ollama model with streaming enabled.
- Validate final citation markers against real source paths and page numbers.
- Show explicit errors for incomplete streams or invalid/missing citations.
- Show tentative output during generation, clear failed drafts, and render final
  Markdown and Persian text direction after validation.
- Support follow-up conversations using role-based Ollama chat messages and fresh
  retrieval for every question. Carry the recent topic into common short follow-ups.
- Supply at most four completed prior turns within a 10,000-character history
  budget, bounded fresh passages, and a 16,384-token chat context window.
- Treat previous answers as conversation context rather than source evidence;
  validate citation IDs against the current turn's retrieved passages only.

## Native interface and lifecycle

- Provide a folder picker, library readiness/counts/progress, mode selector,
  question input, streamed answer, cancellation and PDF/page source list.
- Keep a transcript with saved-chat selection and New chat. Persist completed
  turns and per-answer sources in a separate chats.sqlite database scoped to the
  document index and material folder, surviving index replacement.
- Save only after successful worker completion. Failed/cancelled drafts leave
  prior history intact and never enter the model's conversation context.
- Startup checks readiness; library preparation starts on Update library.
- Keep model and OCR work off the Qt UI thread.
- Run study generation in an owned child process with JSON pipes, not HTTP UI
  endpoints; pass the question through stdin rather than shell arguments.
- Use window-close events to cancel preparation and stop owned processes.
- Release the embedding child runtime on completion/cancellation. Best-effort
  shutdown unloads only the configured generation model after a study request.
- Preserve the existing pre-OCR Ollama unloading policy for GPU availability.
- Keep source/core CLI tools; retire the browser server, static frontend and
  browser-session lifecycle helpers.

## Verification boundaries

- Automated tests cover incremental reuse, activation failures, native/OCR
  routing, streaming protocol, citation errors and Qt worker/lifecycle behavior.
- GUI/model fixtures do not establish live OCR accuracy, GPU throughput or real
  Qwen answer quality. Those require a separately controlled runtime check.
