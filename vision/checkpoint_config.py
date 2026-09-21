"""Durable active-checkpoint pointer for the console and pipeline.

The Flask admin page writes ``data/active_checkpoint.json`` (path configurable).
``create_adapter`` / the app factory read that file when present; otherwise they
fall back to ``ACTIVITY_CHECKPOINT`` / the bundled demo joblib.

This file is not a secret. It only stores a relative or absolute path to a
``.joblib`` under ``data/checkpoints/``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

DEFAULT_ACTIVE_REL = Path("data/active_checkpoint.json")
DEFAULT_CHECKPOINTS_REL = Path("data/checkpoints")
DEFAULT_CHECKPOINT_NAME = "activity_demo.joblib"
ALLOWED_CHECKPOINT_SUFFIXES = {".joblib", ".pkl"}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def active_checkpoint_file(root: str | Path | None = None) -> Path:
    env = os.getenv("ACTIVE_CHECKPOINT_FILE", "").strip()
    if env:
        path = Path(env)
        return path if path.is_absolute() else (root or project_root()) / path
    return (Path(root) if root else project_root()) / DEFAULT_ACTIVE_REL


def checkpoints_dir(root: str | Path | None = None) -> Path:
    env = os.getenv("CHECKPOINTS_DIR", "").strip()
    if env:
        path = Path(env)
        return path if path.is_absolute() else (root or project_root()) / path
    return (Path(root) if root else project_root()) / DEFAULT_CHECKPOINTS_REL


def read_active_checkpoint(path: str | Path | None = None) -> Optional[dict[str, Any]]:
    dest = Path(path) if path is not None else active_checkpoint_file()
    if not dest.is_file():
        return None
    try:
        payload = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict) or not payload.get("path"):
        return None
    return payload


def write_active_checkpoint(
    checkpoint_path: str | Path,
    dest: str | Path | None = None,
    *,
    actor: str = "",
    source: str = "console",
) -> Path:
    dest_path = Path(dest) if dest is not None else active_checkpoint_file()
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "path": str(checkpoint_path),
        "activated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "activated_by": actor,
        "source": source,
    }
    dest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest_path


def _as_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def resolve_checkpoint_path(
    *,
    root: str | Path | None = None,
    active_file: str | Path | None = None,
    env_fallback: str | None = None,
) -> Path:
    """Prefer ``active_checkpoint.json``, then env, then the bundled demo file."""
    root_path = Path(root) if root else project_root()
    record = read_active_checkpoint(active_file or active_checkpoint_file(root_path))
    if record and record.get("path"):
        return _as_path(str(record["path"]), root_path)
    env = (env_fallback if env_fallback is not None else os.getenv("ACTIVITY_CHECKPOINT", "")).strip()
    if env:
        return _as_path(env, root_path)
    return root_path / DEFAULT_CHECKPOINTS_REL / DEFAULT_CHECKPOINT_NAME


def safe_checkpoint_filename(name: str, default: str = "activity_custom.joblib") -> str:
    raw = (name or "").strip() or default
    filename = Path(raw).name
    if not filename or filename in {".", ".."}:
        raise ValueError("Invalid checkpoint filename.")
    if Path(filename).suffix.lower() not in ALLOWED_CHECKPOINT_SUFFIXES:
        filename = f"{filename}.joblib"
    return filename


def resolve_under_checkpoints(
    user_value: str,
    checkpoints: str | Path,
    *,
    default_name: str = "activity_custom.joblib",
) -> Path:
    """Resolve a user-supplied name to a file under the checkpoints directory."""
    folder = Path(checkpoints).resolve()
    filename = safe_checkpoint_filename(user_value, default=default_name)
    dest = (folder / filename).resolve()
    if dest != folder and folder not in dest.parents:
        raise ValueError("Checkpoint path must stay under the checkpoints directory.")
    return dest
