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
SKIP_FILE_NAMES = {".env", ".coverage", "install.log", "mun-cyber-eye-standalone.zip"}
SKIP_SUFFIXES = {".pyc", ".pyo", ".lnk"}

# Files a customer unzip needs in order to install and to launch from the icon.
REQUIRED_ZIP_PATHS = (
    "install_and_run.bat",
    "install_and_run.ps1",
    "Start Mun Cyber Eye.bat",
    "start_eye.ps1",
    "scripts/standalone_support.py",
    "scripts/build_icon.py",
    "app/static/img/mun-cyber-eye.ico",
    "app/static/img/mun-cyber-eye-logo.png",
    "requirements.txt",
    "run.py",
    ".env.example",
    "docs/STANDALONE.md",
)


def should_skip(path: Path, root: Path = ROOT) -> bool:
    rel_parts = path.relative_to(root).parts
    if any(part in SKIP_DIR_NAMES for part in rel_parts):
        return True
    if path.name in SKIP_FILE_NAMES:
        return True
    if path.suffix in SKIP_SUFFIXES:
        return True
    return False


def iter_package_files(root: Path) -> list[Path]:
    files = [path for path in root.rglob("*") if path.is_file() and not should_skip(path, root)]
    return sorted(files)


def missing_required(root: Path) -> list[str]:
    return [rel for rel in REQUIRED_ZIP_PATHS if not (root / rel).is_file()]


def build_zip(root: Path, dest: Path) -> int:
    missing = missing_required(root)
    if missing:
        raise SystemExit("Standalone zip is missing required files: " + ", ".join(missing))
    dest = dest.resolve()
    files = [path for path in iter_package_files(root) if path.resolve() != dest]
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            zf.write(path, path.relative_to(root).as_posix())
    return len(files)


def main() -> None:
    count = build_zip(ROOT, OUT)
    print(f"Wrote {OUT} ({count} files)")
    print("Included the icon, Start Mun Cyber Eye launcher, and install scripts")
    print("Excluded .venv, __pycache__, .git, real .env, install.log, and .lnk shortcuts")


if __name__ == "__main__":
    main()
