from studyrag.ingestion.ocr.paddle_vl import PaddleOcrVlEngine


class FakeResult:
    json = {
        "res": {
            "parsing_res_list": [
                {
                    "block_label": "display_formula",
                    "block_content": "Sp² = 1376.41",
                    "block_bbox": [828, 1520, 1349, 1595],
                    "block_order": 2,
                },
                {
                    "block_label": "text",
                    "block_content": "میانگین وزنی",
                    "block_bbox": [826, 406, 1397, 556],
                    "block_order": 1,
                },
            ]
        }
    }


def test_paddle_result_preserves_layout_blocks_and_reading_order() -> None:
    result = PaddleOcrVlEngine()._parse_result(FakeResult(), ("fas", "eng", "deu"))

    assert result.engine == "paddleocr-vl-1.6"
    assert result.text.startswith("میانگین وزنی")
    assert result.blocks[0].label == "display_formula"
    assert result.blocks[0].bbox == (828, 1520, 1349, 1595)
