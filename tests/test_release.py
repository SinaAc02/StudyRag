import importlib.util
from pathlib import Path
import shutil
import subprocess
import zipfile

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_release.py"
spec = importlib.util.spec_from_file_location("release_check", SCRIPT)
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


def project(tmp_path):
    for name in ("README.md", "pyproject.toml", "StudyRag.cmd", ".gitignore"):
        (tmp_path / name).write_text("Example source\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/app.py").write_text("print('hello')\n", encoding="utf-8")
    return tmp_path


def test_public_source_set_excludes_private_working_directories(tmp_path):
    root = project(tmp_path)
    for name in ("data", "material", "models", ".venv", "work"):
        directory = root / name
        directory.mkdir()
        (directory / "private.txt").write_text("Private fixture", encoding="utf-8")
    files, problems = release.audit(root)
    assert not problems
    assert {path.relative_to(root).as_posix() for path in files} == {
        "README.md",
        "pyproject.toml",
        "StudyRag.cmd",
        ".gitignore",
        "src/app.py",
    }


def test_credentials_are_reported_by_location_without_disclosing_value(tmp_path):
    root = project(tmp_path)
    token = "ghp_" + "x" * 36
    (root / "src/app.py").write_text(f"TOKEN = '{token}'", encoding="utf-8")
    _, problems = release.audit(root)
    assert len(problems) == 1
    assert "src/app.py:1" in problems[0]
    assert token not in problems[0]


def test_private_pdf_in_source_directory_fails_audit(tmp_path):
    root = project(tmp_path)
    (root / "src/private.pdf").write_bytes(b"private fixture")
    _, problems = release.audit(root)
    assert any("private.pdf" in problem for problem in problems)


def test_archive_contains_only_audited_sources(tmp_path, monkeypatch):
    root = project(tmp_path)
    output = tmp_path / "dist/source.zip"
    monkeypatch.setattr("sys.argv", [str(SCRIPT), "--root", str(root), "--archive", str(output)])
    assert release.main() == 0
    with zipfile.ZipFile(output) as archive:
        assert "StudyRag/src/app.py" in archive.namelist()
        assert all(name.startswith("StudyRag/") for name in archive.namelist())


def test_unreviewed_root_file_requires_attention(tmp_path):
    root = project(tmp_path)
    (root / "unreviewed.json").write_text("{}", encoding="utf-8")
    _, problems = release.audit(root)
    assert any("unreviewed.json" in problem for problem in problems)


def test_generated_package_metadata_is_not_exported(tmp_path):
    root = project(tmp_path)
    metadata = root / "src/example.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_text("Generated metadata", encoding="utf-8")
    files, problems = release.audit(root)
    assert not problems
    assert not any("egg-info" in str(path) for path in files)


def test_git_tracked_private_data_fails_even_if_source_export_excludes_it(tmp_path):
    if not shutil.which("git"):
        pytest.skip("Git is unavailable")
    root = project(tmp_path)
    private = root / "data/private.pdf"
    private.parent.mkdir()
    private.write_bytes(b"Synthetic private fixture")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "-f", "data/private.pdf"], check=True)
    _, problems = release.audit(root)
    assert any("private/generated file is tracked" in problem for problem in problems)
