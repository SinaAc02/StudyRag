from types import SimpleNamespace
import sys

import pytest

from studyrag.ingestion.ocr.base import OcrUnavailableError
from studyrag.ingestion.ocr.paddle_structure import (
    PaddleStructureEngine,
    _ScriptRecognizer,
    _ScriptAwareLayout,
)


def test_structure_keeps_table_html_formulas_and_reading_order():
    result = SimpleNamespace(
        json={
            "res": {
                "parsing_res_list": [
                    {
                        "block_label": "table",
                        "block_content": "<table><tr><td>42</td></tr></table>",
                        "block_bbox": [0, 20, 200, 80],
                        "block_order": 2,
                    },
                    {
                        "block_label": "formula",
                        "block_content": r"E=mc^{2}",
                        "block_bbox": [0, 0, 200, 20],
                        "block_order": 1,
                    },
                ]
            }
        }
    )
    parsed = PaddleStructureEngine()._parse_result(result, ("fas", "eng", "deu"))
    assert parsed.text.startswith(r"E=mc^{2}")
    assert "<td>42</td>" in parsed.text
    assert parsed.blocks[0].label == "table"
    assert parsed.blocks[1].bbox == (0, 0, 200, 20)


def test_script_recognizer_shares_crops_and_selects_each_line():
    def latin(images, **kwargs):
        assert kwargs == {"return_word_box": False}
        return [{"rec_text": "Straße", "rec_score": 0.95 if images[0] == 1 else 0.1}]

    def arabic(images, **kwargs):
        return [{"rec_text": "میانگین", "rec_score": 0.97 if images[0] == 2 else 0.1}]

    results = list(_ScriptRecognizer(latin, arabic)([1, 2], return_word_box=False))
    assert [result["rec_text"] for result in results] == ["Straße", "میانگین"]


def test_script_recognizer_rejects_missing_results():
    with pytest.raises(OcrUnavailableError, match="line count"):
        list(_ScriptRecognizer(lambda *a, **k: [], lambda *a, **k: [])([1]))


def test_config_bounds_all_batches_and_retains_table_formula_models(monkeypatch):
    original = {
        "batch_size": 8,
        "SubPipelines": {
            "GeneralOCR": {
                "batch_size": 8,
                "SubModules": {"TextDetection": {}, "TextRecognition": {"batch_size": 8}},
            },
            "FormulaRecognition": {"SubModules": {"FormulaRecognition": {"batch_size": 8}}},
            "TableRecognition": {"batch_size": 8},
        },
    }
    monkeypatch.setitem(
        sys.modules, "paddlex.inference", SimpleNamespace(load_pipeline_config=lambda name: original)
    )
    config = PaddleStructureEngine._configuration()
    assert original["batch_size"] == 8
    assert config["batch_size"] == 1
    assert config["use_table_recognition"] and config["use_formula_recognition"]
    formula = config["SubPipelines"]["FormulaRecognition"]["SubModules"]["FormulaRecognition"]
    assert formula["model_name"] == "PP-FormulaNet_plus-L"
    assert formula["batch_size"] == 1
    assert config["SubPipelines"]["GeneralOCR"]["SubModules"]["TextDetection"]["limit_type"] == "max"


def test_image_bound_changes_processing_key():
    assert PaddleStructureEngine(max_image_side=1200).cache_key != PaddleStructureEngine().cache_key


def test_large_page_is_bounded_before_inference(tmp_path, monkeypatch):
    import pymupdf

    path = tmp_path / "large.pdf"
    with pymupdf.open() as document:
        page = document.new_page(width=3000, height=5000)
        page.draw_rect(page.rect, color=(1, 0, 0), fill=(1, 0, 0))
        document.save(path)
    engine = PaddleStructureEngine(max_image_side=1000)
    monkeypatch.setattr(engine, "validate", lambda languages: None)

    def predict(image):
        assert image.shape == (1000, 600, 3)
        assert tuple(image[500, 300]) == (0, 0, 255)  # BGR
        return [SimpleNamespace(json={"res": {"parsing_res_list": []}})]

    monkeypatch.setattr(engine, "_get_pipeline", lambda: SimpleNamespace(predict=predict))
    assert engine.recognize_page(path, 1, ("eng",)).text == ""


def test_inference_error_is_reported_without_cpu_retry(tmp_path, monkeypatch):
    import pymupdf

    path = tmp_path / "page.pdf"
    with pymupdf.open() as document:
        document.new_page()
        document.save(path)
    engine = PaddleStructureEngine()
    monkeypatch.setattr(engine, "validate", lambda languages: None)

    def fail(image):
        raise RuntimeError("out of memory")

    monkeypatch.setattr(engine, "_get_pipeline", lambda: SimpleNamespace(predict=fail))
    with pytest.raises(OcrUnavailableError, match="No CPU fallback"):
        engine.recognize_page(path, 1, ("eng",))


def test_table_cell_repair_shares_the_multilingual_ocr(monkeypatch):
    latin = object()
    arabic = object()
    internal_ocr = SimpleNamespace(text_rec_model=latin)
    ocr = SimpleNamespace(_pipeline=internal_ocr)
    internal_table = SimpleNamespace(general_ocr_pipeline=None)
    table = SimpleNamespace(_pipeline=internal_table)
    pipeline = SimpleNamespace(
        paddlex_pipeline=SimpleNamespace(
            general_ocr_pipeline=ocr, table_recognition_pipeline=table, layout_det_model=object()
        )
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=lambda **kwargs: pipeline))
    monkeypatch.setitem(sys.modules, "paddlex", SimpleNamespace(create_model=lambda *args, **kwargs: arabic))
    engine = PaddleStructureEngine()
    monkeypatch.setattr(engine, "_configuration", lambda: {})
    assert engine._get_pipeline() is pipeline
    assert internal_table.general_ocr_pipeline is ocr
    assert isinstance(internal_ocr.text_rec_model, _ScriptRecognizer)
    assert internal_ocr.text_rec_model.arabic is arabic


def test_persian_sentence_is_not_sent_to_formula_recognizer():
    import numpy as np

    boxes = [{"label": "formula", "coordinate": [0, 0, 100, 40]}]
    detector = lambda images, **kwargs: iter([{"boxes": boxes}])
    ocr = lambda image: iter([{"rec_texts": ["مایع حاصل از دوش کامل پستان دام"], "rec_scores": [0.96]}])
    result = list(_ScriptAwareLayout(detector, ocr)([np.zeros((100, 100, 3))]))
    assert result[0]["boxes"][0]["label"] == "text"


def test_mathematical_formula_keeps_its_layout_label():
    import numpy as np

    boxes = [{"label": "formula", "coordinate": [0, 0, 100, 40]}]
    detector = lambda images, **kwargs: iter([{"boxes": boxes}])
    ocr = lambda image: iter([{"rec_texts": ["x = b2 - 4ac"], "rec_scores": [0.92]}])
    result = list(_ScriptAwareLayout(detector, ocr)([np.zeros((100, 100, 3))]))
    assert result[0]["boxes"][0]["label"] == "formula"
