#!/usr/bin/env python3
"""Build mun-cyber-eye-standalone.zip without .venv, caches, git, or real .env.

Usage (from repo root):
    python scripts/build_standalone_zip.py
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "mun-cyber-eye-standalone.zip"

SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "htmlcov",
    "ultralytics",
    "runs",
    ".cursor",
}
SKIP_FILE_NAMES = {".env", ".coverage"}
SKIP_SUFFIXES = {".pyc", ".pyo"}


def should_skip(path: Path) -> bool:
    rel_parts = path.relative_to(ROOT).parts
    if any(part in SKIP_DIR_NAMES for part in rel_parts):
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def main() -> None:
    files = [p for p in ROOT.rglob("*") if p.is_file() and not should_skip(p)]
    if OUT in files:
        files.remove(OUT)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(files):
            zf.write(path, path.relative_to(ROOT).as_posix())
    print(f"Wrote {OUT} ({len(files)} files)")
    print("Excluded .venv, __pycache__, .git, and real .env")


if __name__ == "__main__":
    main()
