# Third-party dependencies and licensing

StudyRag's own source is currently **unlicensed**. Dependencies and downloaded
models have separate license terms. Source releases contain the application
code and documentation; dependencies and model weights are installed separately.

- **PyMuPDF / MuPDF:** dual-licensed under GNU AGPL v3 and an Artifex commercial
  license. Review the applicable terms before distributing or deploying an
  application that uses them. These terms also apply when the application's own
  source is unlicensed.
  [PyMuPDF licensing](https://pymupdf.io/licensing)
- **PySide6 / Qt for Python:** review the applicable Qt community or commercial
  license terms and third-party notices before redistributing Qt binaries.
  [Qt for Python licenses](https://doc.qt.io/qtforpython-6/licenses.html)
- **PaddleOCR / PaddleX / PaddlePaddle:** use their published license and model
  terms. [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR),
  [PaddleX](https://github.com/PaddlePaddle/PaddleX),
  [PaddlePaddle](https://github.com/PaddlePaddle/Paddle)
- **NumPy, sentence-transformers, Transformers and PyTorch:** retain the notices
  supplied by each dependency when redistribution requires it.
- **Downloaded model weights:** review the original model's terms. Ollama is a
  runtime; each model has its own license. StudyRag's configured models
  are PP-StructureV3 components, PP-FormulaNet_plus-L, multilingual-e5-small and
  the locally installed qwen3.5:9b Ollama model.

Consult the linked projects for their full license texts and notices when
preparing a distribution. A bundled binary release would need its own review.
