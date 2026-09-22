#!/usr/bin/env python3
"""Stdlib helpers for the Windows standalone installer.

``install_and_run.ps1`` calls this with system Python before ``.venv``
packages exist. The script decides whether the environment can start,
needs ``requirements.txt``, should drop optional torch leftovers, or
should be recreated.

Pip renames a distribution by replacing its first character with ``~``
while upgrading. An interrupted torch install leaves ``~orch``. An
interrupted ultralytics install leaves ``~ltralytics``. Those leftovers
are optional and must not fail the core Flask / OpenCV install.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

CRITICAL_IMPORT_STATEMENT = "import flask, cv2, numpy, PIL, sklearn, dotenv, joblib"

# Pip's in-progress rename drops the first character and prefixes "~".
OPTIONAL_ML_PACKAGES = ("torch", "torchvision", "torchaudio", "ultralytics")


def venv_python(venv: Path) -> Path | None:
    """Return the venv interpreter, Windows layout first."""
    candidates = (
        venv / "Scripts" / "python.exe",
        venv / "Scripts" / "python",
        venv / "bin" / "python",
        venv / "bin" / "python.exe",
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def site_packages_dirs(venv: Path) -> list[Path]:
    found: list[Path] = []
    windows = venv / "Lib" / "site-packages"
    if windows.is_dir():
        found.append(windows)
    lib = venv / "lib"
    if lib.is_dir():
        found.extend(sorted(path for path in lib.glob("python*/site-packages") if path.is_dir()))
    return found


def is_optional_ml_name(name: str) -> bool:
    """True for torch/ultralytics names and their ``~orch`` / ``~ltralytics`` leftovers."""
    lowered = name.lower()
    if lowered.startswith("~"):
        tail = lowered[1:]
        for base in OPTIONAL_ML_PACKAGES:
            rest = base[1:]
            if tail == rest or tail.startswith(rest + "-") or tail.startswith(rest + "."):
                return True
        return False
    stem = lowered
    for suffix in (".dist-info", ".egg-info"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    stem = stem.split("-", 1)[0]
    return stem in OPTIONAL_ML_PACKAGES


def distribution_marker(entry: Path) -> str | None:
    """Name of a pip leftover or a distribution directory missing its metadata."""
    name = entry.name
    if name.startswith("~"):
        return name
    if name.endswith(".dist-info") or name.endswith(".egg-info"):
        if not entry.is_dir():
            return name
        if not (entry / "METADATA").is_file() and not (entry / "PKG-INFO").is_file():
            return name
    return None


def collect_markers(venv: Path) -> list[str]:
    found: list[str] = []
    for site in site_packages_dirs(venv):
        for entry in site.iterdir():
            marker = distribution_marker(entry)
            if marker:
                found.append(marker)
    return sorted(set(found))


def cloud_locked_install_path(install_path: str, desktop_path: str | None = None) -> bool:
    """True when the folder is under OneDrive, including a redirected Desktop."""
    full = install_path.replace("/", "\\")
    if "onedrive" in full.lower():
        return True
    if not desktop_path:
        return False
    desktop = desktop_path.replace("/", "\\").rstrip("\\")
    if "onedrive" not in desktop.lower():
        return False
    prefix = desktop.lower()
    candidate = full.rstrip("\\").lower()
    return candidate == prefix or candidate.startswith(prefix + "\\")


def _run_python(python: Path, code: str) -> int | None:
    try:
        completed = subprocess.run(
            [str(python), "-c", code],
            capture_output=True,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return completed.returncode


def assess_venv(venv: Path) -> dict[str, str]:
    """Choose ready, install, scrub_optional, or recreate.

    Every value is a string so Windows PowerShell 5.1 can read the JSON
    without turning empty arrays into null.
    """
    python = venv_python(venv)
    markers = collect_markers(venv) if venv.exists() else []
    optional = [name for name in markers if is_optional_ml_name(name)]
    core_markers = [name for name in markers if name not in optional]
    marker_text = ", ".join(markers)
    interpreter_ok = python is not None and _run_python(python, "import sys") == 0

    if not interpreter_ok:
        if python is None:
            detail = "No .venv Python was found."
            reason = "missing"
        else:
            detail = "The .venv Python does not run."
            if marker_text:
                detail += " Interrupted or invalid packages: " + marker_text
            reason = "damaged"
        return {
            "action": "recreate",
            "reason": reason,
            "detail": detail,
            "markers": marker_text,
        }

    imports_ok = _run_python(python, CRITICAL_IMPORT_STATEMENT) == 0
    if core_markers or (optional and not imports_ok):
        detail = "Interrupted or invalid packages: " + marker_text
        return {
            "action": "recreate",
            "reason": "damaged",
            "detail": detail,
            "markers": marker_text,
        }
    if not imports_ok:
        return {
            "action": "install",
            "reason": "incomplete",
            "detail": "Core packages are missing (flask, cv2, numpy, Pillow, scikit-learn, python-dotenv, joblib).",
            "markers": marker_text,
        }
    if optional:
        return {
            "action": "scrub_optional",
            "reason": "optional_leftovers",
            "detail": "Optional ultralytics/torch leftovers can be removed: " + ", ".join(optional),
            "markers": ", ".join(optional),
        }
    return {
        "action": "ready",
        "reason": "ready",
        "detail": "Core packages import.",
        "markers": "",
    }


def optional_marker_paths(venv: Path) -> list[Path]:
    paths: list[Path] = []
    for site in site_packages_dirs(venv):
        for entry in site.iterdir():
            marker = distribution_marker(entry)
            if marker and is_optional_ml_name(marker):
                paths.append(entry)
    return paths


def scrub_optional(venv: Path) -> list[str]:
    """Delete optional torch/ultralytics leftovers. Leave core packages in place."""
    removed: list[str] = []
    for path in optional_marker_paths(venv):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        removed.append(path.name)
    return sorted(set(removed))


def _print_json(payload: dict[str, str]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Mun Cyber Eye standalone install helpers")
    sub = parser.add_subparsers(dest="cmd", required=True)

    assess = sub.add_parser("assess", help="Print JSON: action, reason, detail, markers")
    assess.add_argument("venv")

    scrub = sub.add_parser("scrub-optional", help="Remove optional torch/ultralytics leftovers")
    scrub.add_argument("venv")

    risk = sub.add_parser("path-risk", help="Print risky or ok for OneDrive/Desktop installs")
    risk.add_argument("path")
    risk.add_argument("--desktop", default="")

    args = parser.parse_args(argv)
    if args.cmd == "assess":
        _print_json(assess_venv(Path(args.venv)))
        return 0
    if args.cmd == "scrub-optional":
        removed = scrub_optional(Path(args.venv))
        _print_json({"removed": ", ".join(removed)})
        return 0
    if args.cmd == "path-risk":
        risky = cloud_locked_install_path(args.path, args.desktop or None)
        sys.stdout.write("risky\n" if risky else "ok\n")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
