from __future__ import annotations

import copy
import os
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf

from studyrag.domain.models import OcrBlock, OcrResult
from studyrag.ingestion.ocr.base import OcrUnavailableError
from studyrag.settings import PROJECT_ROOT


class _ScriptRecognizer:
    """Recognize each line in both scripts; confidence selection is a heuristic."""

    def __init__(self, latin: Any, arabic: Any) -> None:
        self.latin = latin
        self.arabic = arabic

    def __getattr__(self, name: str) -> Any:
        return getattr(self.latin, name)

    def __call__(self, images: Any, **kwargs: Any):
        # Batch size 1 limits peak memory for each recognizer.
        for image in images:
            latin = list(self.latin([image], **kwargs))
            arabic = list(self.arabic([image], **kwargs))
            if len(latin) != 1 or len(arabic) != 1:
                raise OcrUnavailableError("OCR recognizer returned an unexpected line count.")
            yield max((latin[0], arabic[0]), key=lambda result: result["rec_score"])


class _ScriptAwareLayout:
    """Reclassify confident Persian sentence detections as text."""

    def __init__(self, detector: Any, ocr: Any) -> None:
        self.detector = detector
        self.ocr = ocr

    def __getattr__(self, name: str) -> Any:
        return getattr(self.detector, name)

    def __call__(self, images: Any, **kwargs: Any):
        for image, result in zip(images, self.detector(images, **kwargs), strict=True):
            for box in result["boxes"]:
                if box["label"] != "formula":
                    continue
                x1, y1, x2, y2 = (int(value) for value in box["coordinate"])
                crop = image[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
                if not crop.size:
                    continue
                for recognized in self.ocr(crop):
                    lines = zip(recognized["rec_texts"], recognized["rec_scores"], strict=True)
                    for text, score in lines:
                        letters = [character for character in text if character.isalpha()]
                        arabic = sum("ARABIC" in unicodedata.name(character, "") for character in letters)
                        if score >= 0.7 and arabic >= 8 and arabic >= len(letters) * 0.5:
                            box.update(label="text", cls_id=2)  # PP-DocLayout_plus-L text class
                            break
                    if box["label"] == "text":
                        break
            yield result


class PaddleStructureEngine:
    """Bounded PP-StructureV3 extraction with text, tables and LaTeX formulas."""

    name = "pp-structure-v3-mobile-formula-l"

    def __init__(
        self,
        *,
        device: str = "gpu:0",
        render_scale: float = 2.5,
        max_image_side: int = 2000,
    ) -> None:
        self.device = device
        self.render_scale = render_scale
        self.max_image_side = max_image_side
        self._pipeline: Any | None = None

    @property
    def cache_key(self) -> str:
        return (
            f"{self.name}:scale={self.render_scale}:max_side={self.max_image_side}"
            ":batch=1:latin-arabic-confidence-v1:tables:formula:policy=2"
        )

    def validate(self, languages: tuple[str, ...]) -> None:
        if set(languages) - {"fas", "eng", "deu"}:
            raise OcrUnavailableError("The OCR backend supports Persian, English and German.")
        # Paddle reads these settings on import in the OCR child process.
        os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(PROJECT_ROOT / "models/paddlex"))
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        os.environ.setdefault("FLAGS_allocator_strategy", "auto_growth")
        os.environ.setdefault("FLAGS_gpu_memory_limit_mb", "6144")
        try:
            import torch  # noqa: F401 -- load first to avoid Windows CUDA DLL collisions
            import paddle
            from paddleocr import PPStructureV3  # noqa: F401
        except ImportError as error:
            raise OcrUnavailableError(
                "PP-StructureV3 requires paddleocr[doc-parser] and PaddlePaddle."
            ) from error
        if self.device.startswith("gpu") and not paddle.device.is_compiled_with_cuda():
            raise OcrUnavailableError("OCR requires CUDA PaddlePaddle; CPU fallback is disabled.")

    @staticmethod
    def _configuration() -> dict:
        from paddlex.inference import load_pipeline_config

        config = copy.deepcopy(load_pipeline_config("PP-StructureV3"))

        def bound_batches(value: Any) -> None:
            if isinstance(value, dict):
                if "batch_size" in value:
                    value["batch_size"] = 1
                if "pipeline_name" in value:
                    value.update(
                        use_doc_preprocessor=False,
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                    )
                for child in value.values():
                    bound_batches(child)

        bound_batches(config)
        config.update(
            use_doc_preprocessor=False,
            use_seal_recognition=False,
            use_chart_recognition=False,
            use_region_detection=False,
            use_table_recognition=True,
            use_formula_recognition=True,
        )
        ocr = config["SubPipelines"]["GeneralOCR"]
        ocr["use_textline_orientation"] = False
        ocr["SubModules"]["TextDetection"].update(
            model_name="PP-OCRv5_mobile_det",
            limit_side_len=1536,
            limit_type="max",
            max_side_limit=2000,
        )
        ocr["SubModules"]["TextRecognition"]["model_name"] = "latin_PP-OCRv5_mobile_rec"
        config["SubPipelines"]["FormulaRecognition"]["SubModules"]["FormulaRecognition"]["model_name"] = (
            "PP-FormulaNet_plus-L"
        )
        return config

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from paddleocr import PPStructureV3
            from paddlex import create_model

            pipeline = PPStructureV3(
                paddlex_config=self._configuration(),
                device=self.device,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
            # PaddleOCR 3.7 has one recognizer slot. Both scripts share the
            # detector, layout, table and formula models. These internal slots
            # match the PaddleOCR 3.7 range in pyproject.toml.
            ocr = pipeline.paddlex_pipeline.general_ocr_pipeline
            internal_ocr = getattr(ocr, "_pipeline", ocr)
            latin = internal_ocr.text_rec_model
            arabic = create_model("arabic_PP-OCRv5_mobile_rec", device=self.device, batch_size=1)
            internal_ocr.text_rec_model = _ScriptRecognizer(latin, arabic)
            # Sharing the bilingual pipeline also covers table cell repair,
            # whose default is a separate Chinese/English server OCR pipeline.
            table = pipeline.paddlex_pipeline.table_recognition_pipeline
            internal_table = getattr(table, "_pipeline", table)
            internal_table.general_ocr_pipeline = ocr
            structure = pipeline.paddlex_pipeline
            internal_structure = getattr(structure, "_pipeline", structure)
            internal_structure.layout_det_model = _ScriptAwareLayout(
                internal_structure.layout_det_model,
                ocr,
            )
            self._pipeline = pipeline
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
                scale = min(self.render_scale, self.max_image_side / max(page.rect.width, page.rect.height))
                pixmap = page.get_pixmap(
                    matrix=pymupdf.Matrix(scale, scale),
                    alpha=False,
                    colorspace=pymupdf.csRGB,
                )
                rgb = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width, 3)
                # NumPy inputs to Paddle's conventional OCR pipeline are BGR.
                image = np.ascontiguousarray(rgb[:, :, ::-1])
            results = list(self._get_pipeline().predict(image))
            if len(results) != 1:
                raise ValueError("Expected exactly one page result.")
            return self._parse_result(results[0], languages)
        except OcrUnavailableError:
            raise
        except Exception as error:
            raise OcrUnavailableError(
                f"PP-StructureV3 failed on {pdf_path.name}, page {page_number}: {error}. "
                "No CPU fallback was attempted."
            ) from error

    def _parse_result(self, result: Any, languages: tuple[str, ...]) -> OcrResult:
        payload = result.json
        payload = payload() if callable(payload) else payload
        root = payload.get("res", payload)
        blocks = tuple(
            OcrBlock(
                str(item.get("block_label", "text")),
                str(item.get("block_content", "")).strip(),
                tuple(int(value) for value in item["block_bbox"]),
                item.get("block_order"),
            )
            for item in root.get("parsing_res_list", ())
            if str(item.get("block_content", "")).strip() and len(item.get("block_bbox", ())) == 4
        )
        ordered = sorted(blocks, key=lambda block: (block.order is None, block.order or 0))
        return OcrResult("\n\n".join(block.text for block in ordered), self.name, languages, blocks)
