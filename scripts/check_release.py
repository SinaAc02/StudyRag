"""Audit public source files and export a source archive."""

import argparse
from pathlib import Path
import re
import subprocess
import sys
import zipfile


ROOT_FILES = {
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "README.md",
    "SPEC.md",
    "StudyRag.cmd",
    "THIRD_PARTY_NOTICES.md",
    "requirements-test.txt",
    "MANIFEST.in",
    "LICENSE",
    "LICENSE.md",
    "NOTICE",
    "CHANGELOG.md",
}
SOURCE_DIRECTORIES = {"src", "tests", "scripts", ".github", "docs"}
DOCUMENTATION_IMAGES = {"docs/images/studyrag.png"}
LOCAL_DIRECTORIES = {
    ".git",
    ".venv",
    ".test-venv",
    "venv",
    "env",
    "work",
    "data",
    "material",
    "models",
    "graphify-out",
    "build",
    "dist",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    ".vscode",
    ".idea",
    "htmlcov",
}
PRIVATE_SUFFIXES = {
    ".pdf",
    ".sqlite",
    ".db",
    ".log",
    ".gguf",
    ".safetensors",
    ".pdparams",
    ".pdiparams",
    ".onnx",
    ".pem",
    ".key",
    ".p12",
    ".pfx",
}
SECRET_PATTERNS = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{50,255})\b"),
    "AWS access key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "API token": re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
}


def source_files(root: Path) -> list[Path]:
    root = root.resolve()
    files = [root / name for name in ROOT_FILES if (root / name).is_file()]
    for directory in sorted(SOURCE_DIRECTORIES):
        if not (root / directory).is_dir():
            continue
        for path in (root / directory).rglob("*"):
            if (
                "__pycache__" in path.parts
                or path.suffix in {".pyc", ".pyo"}
                or any(part.endswith(".egg-info") for part in path.relative_to(root).parts)
            ):
                continue
            if path.is_file():
                files.append(path)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def audit(root: Path) -> tuple[list[Path], list[str]]:
    root = root.resolve()
    files = source_files(root)
    problems = []
    for entry in root.iterdir():
        if entry.is_dir():
            if entry.name not in SOURCE_DIRECTORIES | LOCAL_DIRECTORIES and not entry.name.endswith(
                ".egg-info"
            ):
                problems.append(f"{entry.name}/: unreviewed top-level directory")
        elif entry.name not in ROOT_FILES:
            if (
                entry.suffix.lower() not in PRIVATE_SUFFIXES
                and not entry.name.startswith(".env")
                and entry.name not in {".coverage", "coverage.xml", "Thumbs.db", ".DS_Store"}
            ):
                problems.append(f"{entry.name}: unreviewed top-level file")
    # Tracked files need a separate check because Git ignores apply to untracked files.
    if (root / ".git").exists():
        try:
            tracked = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"])
            for name in tracked.decode("utf-8").split("\0"):
                if not name:
                    continue
                path = Path(name)
                if (
                    path.parts[0] in LOCAL_DIRECTORIES
                    or path.suffix.lower() in PRIVATE_SUFFIXES
                    or path.name.startswith(".env")
                    and path.name != ".env.example"
                ):
                    problems.append(f"{name}: private/generated file is tracked by Git")
        except (OSError, subprocess.CalledProcessError, UnicodeError):
            problems.append("Could not inspect tracked Git files")
    for path in files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            problems.append(f"{relative}: source exports must not contain symlinks")
            continue
        if path.suffix.lower() in PRIVATE_SUFFIXES or path.name.startswith(".env"):
            problems.append(f"{relative}: private data/credential/model file in public source set")
            continue
        if path.stat().st_size > 2_000_000:
            problems.append(f"{relative}: unexpectedly large source file")
            continue
        if relative in DOCUMENTATION_IMAGES:
            with path.open("rb") as image:
                if image.read(8) != b"\x89PNG\r\n\x1a\n":
                    problems.append(f"{relative}: invalid PNG image")
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeError:
            problems.append(f"{relative}: binary or non-UTF-8 source file")
            continue
        for label, pattern in SECRET_PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                # Credential findings contain locations and types, with values omitted.
                problems.append(f"{relative}:{line}: possible {label}")
    for required in ("README.md", "pyproject.toml", "StudyRag.cmd", ".gitignore"):
        if not (root / required).is_file():
            problems.append(f"Missing required source file: {required}")
    return files, problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, help="Export the audited source set to a NEW zip file")
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    files, problems = audit(root)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"Source audit passed: {len(files)} files, {sum(p.stat().st_size for p in files):,} bytes.")
    print("Private PDFs, local data, models, virtual environments and work directories are excluded.")
    if arguments.archive:
        archive = arguments.archive.resolve()
        archive.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects existing archives.
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as output:
            for path in files:
                output.write(path, Path("StudyRag") / path.relative_to(root))
        print(f"Source archive: {archive}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
