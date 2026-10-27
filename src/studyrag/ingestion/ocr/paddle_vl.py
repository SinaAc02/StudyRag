from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf
import numpy as np

from studyrag.domain.models import OcrBlock, OcrResult
from studyrag.ingestion.ocr.base import OcrUnavailableError


class PaddleOcrVlEngine:
    """Layout-aware PaddleOCR-VL 1.6 adapter using the local GPU."""

    name = "paddleocr-vl-1.6"

    def __init__(self, *, device: str = "gpu:0", render_scale: float = 2.5) -> None:
        self.device = device
        self.render_scale = render_scale
        self._pipeline: Any | None = None

    @property
    def cache_key(self) -> str:
        return f"{self.name}:render_scale={self.render_scale}:layout:charts:image-text"

    def validate(self, languages: tuple[str, ...]) -> None:
        del languages  # PaddleOCR-VL is multilingual and detects language itself.
        try:
            # PaddleX imports ModelScope, which imports Torch. On Windows, loading
            # CPU Torch after CUDA Paddle can produce a DLL collision. Loading
            # Torch first establishes its DLLs before Paddle initializes.
            import torch  # noqa: F401
            import paddle
            from paddleocr import PaddleOCRVL  # noqa: F401
        except ImportError as error:
            raise OcrUnavailableError(
                "PaddleOCR-VL is not installed. Install PaddlePaddle and paddleocr[doc-parser]."
            ) from error
        if self.device.startswith("gpu") and not paddle.device.is_compiled_with_cuda():
            raise OcrUnavailableError("PaddleOCR-VL requested a GPU, but Paddle has no CUDA support.")

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PaddleOCRVL

            self._pipeline = PaddleOCRVL(
                pipeline_version="v1.6",
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_layout_detection=True,
                use_chart_recognition=True,
                use_ocr_for_image_block=True,
            )
        return self._pipeline

    def recognize_page(
        self,
        pdf_path: Path,
        page_number: int,
        languages: tuple[str, ...],
    ) -> OcrResult:
        self.validate(languages)
        try:
            with pymupdf.open(pdf_path) as document:
                page = document[page_number - 1]
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(self.render_scale, self.render_scale),
                    alpha=False,
                )
                image = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
                    pixmap.height, pixmap.width, pixmap.n
                )
            results = list(self._get_pipeline().predict(image))
        except Exception as error:
            raise OcrUnavailableError(
                f"PaddleOCR-VL failed on {pdf_path.name}, page {page_number}: {error}"
            ) from error
        if not results:
            raise OcrUnavailableError(
                f"PaddleOCR-VL returned no result for {pdf_path.name}, page {page_number}"
            )

        return self._parse_result(results[0], languages)

    def _parse_result(self, result: Any, languages: tuple[str, ...]) -> OcrResult:
        payload = result.json
        if callable(payload):
            payload = payload()
        root = payload.get("res", payload)
        blocks = tuple(
            OcrBlock(
                label=str(item.get("block_label", "text")),
                text=str(item.get("block_content", "")).strip(),
                bbox=tuple(int(value) for value in item.get("block_bbox", (0, 0, 0, 0))),
                order=item.get("block_order"),
            )
            for item in root.get("parsing_res_list", ())
            if str(item.get("block_content", "")).strip() and len(item.get("block_bbox", ())) == 4
        )
        ordered = sorted(blocks, key=lambda block: (block.order is None, block.order or 0))
        text = "\n\n".join(block.text for block in ordered)
        return OcrResult(text, self.name, languages, blocks)
