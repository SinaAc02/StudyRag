from studyrag import settings
import json
import os
import subprocess
import sys

import pytest


def test_checkout_defaults_to_repository_root(monkeypatch):
    monkeypatch.delenv("STUDYRAG_HOME", raising=False)
    assert (settings.application_home() / "pyproject.toml").is_file()


def test_explicit_home_overrides_checkout(monkeypatch, tmp_path):
    monkeypatch.setenv("STUDYRAG_HOME", str(tmp_path / "custom"))
    assert settings.application_home() == (tmp_path / "custom").resolve()


def test_installed_windows_build_uses_local_app_data(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYRAG_HOME", raising=False)
    monkeypatch.setattr(settings, "__file__", str(tmp_path / "site-packages/studyrag/settings.py"))
    monkeypatch.setattr(settings.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "app-data"))
    assert settings.application_home() == (tmp_path / "app-data/StudyRag").resolve()


def test_installed_unix_build_uses_xdg_data_home(monkeypatch, tmp_path):
    monkeypatch.delenv("STUDYRAG_HOME", raising=False)
    monkeypatch.setattr(settings, "__file__", str(tmp_path / "site-packages/studyrag/settings.py"))
    monkeypatch.setattr(settings.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "app-data"))
    assert settings.application_home() == (tmp_path / "app-data/StudyRag").resolve()


def test_worker_starts_from_separate_data_home_without_loading_models(tmp_path):
    environment = dict(
        os.environ, PYTHONPATH=str(settings.PACKAGE_IMPORT_ROOT), STUDYRAG_HOME=str(tmp_path), PYTHONUTF8="1"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-m", "studyrag.desktop.worker"],
        cwd=tmp_path,
        env=environment,
        input=json.dumps({"database": str(tmp_path / "missing.sqlite"), "prompt": "test"}),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
    )
    assert result.returncode == 1
    event = json.loads(result.stdout)
    assert event["type"] == "error" and "database does not exist" in event["message"]


@pytest.mark.parametrize(
    "arguments", [["ingest", "material"], ["index"], ["search", "question"], ["ask", "question"], ["desktop"]]
)
def test_cli_commands_share_the_application_database_default(arguments):
    from studyrag.cli import build_parser

    assert build_parser().parse_args(arguments).database == settings.DEFAULT_DATABASE
