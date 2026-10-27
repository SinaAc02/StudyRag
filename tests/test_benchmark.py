import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_retrieval.py"
spec = importlib.util.spec_from_file_location("retrieval_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


@pytest.mark.parametrize("cases", [[], {}, ["text"], [{"query": "question"}]])
def test_invalid_cases_are_rejected_before_loading_a_model(tmp_path, cases):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases), encoding="utf-8")
    with pytest.raises(ValueError):
        benchmark.load_cases(path)


def test_missing_index_fails_before_loading_a_model(tmp_path, monkeypatch):
    path = tmp_path / "cases.json"
    path.write_text(
        json.dumps([{"query": "question", "expected_filename_contains": "lesson"}]), encoding="utf-8"
    )
    monkeypatch.setattr(
        "sys.argv", [str(SCRIPT), "--database", str(tmp_path / "missing.sqlite"), "--cases", str(path)]
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("Invalid benchmark inputs must not load a model")

    monkeypatch.setattr(benchmark, "E5Embedder", forbidden)
    assert benchmark.main() == 2
