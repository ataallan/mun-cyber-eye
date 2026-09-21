"""Helpers for the Admin → Train models console.

Thin wrappers around ``vision.train_activity`` / ``vision.eval_activity`` plus
safe upload and active-checkpoint activation. Training math stays in vision/.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Iterable

from vision.activity import inspect_checkpoint, load_checkpoint
from vision.checkpoint_config import (
    read_active_checkpoint,
    resolve_under_checkpoints,
    write_active_checkpoint,
)
from vision.dataset import (
    ACTIVITY_CATEGORIES,
    IMAGE_SUFFIXES,
    SPLITS,
    describe_dataset,
    ensure_dataset_tree,
    parse_folder_label,
)
from vision.sports_catalog import resolve_sport
from vision.eval_activity import evaluate_checkpoint
from vision.metrics import format_metrics_report
from vision.train_activity import train_activity_model

DEMO_CHECKPOINT_NAME = "activity_demo.joblib"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_ZIP_MEMBERS = 400


def data_root_from_config(config: dict) -> Path:
    return Path(config["ACTIVITY_DATA_ROOT"])


def checkpoints_dir_from_config(config: dict) -> Path:
    return Path(config["CHECKPOINTS_DIR"])


def active_file_from_config(config: dict) -> Path:
    return Path(config["ACTIVE_CHECKPOINT_FILE"])


def dataset_inventory(config: dict) -> dict:
    root = data_root_from_config(config)
    ensure_dataset_tree(root)
    return describe_dataset(root)


def list_checkpoint_files(config: dict) -> list[Path]:
    folder = checkpoints_dir_from_config(config)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.joblib") if p.is_file())


def current_checkpoint_info(config: dict) -> dict[str, Any]:
    path = Path(config["ACTIVITY_CHECKPOINT"])
    info = inspect_checkpoint(path)
    record = read_active_checkpoint(active_file_from_config(config))
    info["active_pointer"] = str(active_file_from_config(config))
    info["active_record"] = record
    info["source"] = "active_checkpoint.json" if record else "ACTIVITY_CHECKPOINT / default"
    return info


def apply_active_checkpoint(config: dict, checkpoint_path: str | Path, actor: str) -> Path:
    raw = str(checkpoint_path or "").strip()
    if not raw:
        raise ValueError("Choose a checkpoint file.")
    folder = checkpoints_dir_from_config(config).resolve()
    candidate = Path(raw)
    dest = (
        candidate.resolve()
        if candidate.is_absolute()
        else resolve_under_checkpoints(raw, folder, default_name=raw)
    )
    if dest != folder and folder not in dest.parents:
        raise ValueError("Checkpoint must live under the checkpoints directory.")
    if dest.suffix.lower() not in {".joblib", ".pkl"}:
        raise ValueError("Checkpoint must be a .joblib file.")
    if not dest.is_file():
        raise ValueError(f"Checkpoint not found: {dest}")
    load_checkpoint(dest)  # fail closed if the file cannot be loaded
    write_active_checkpoint(
        dest,
        dest=active_file_from_config(config),
        actor=actor,
        source="console",
    )
    config["ACTIVITY_CHECKPOINT"] = str(dest)
    return dest


def run_training(
    config: dict,
    *,
    model_type: str,
    output_name: str,
    generate_demo: bool,
    overwrite_demo_images: bool,
    confirm_overwrite_demo: bool,
    seed: int = 7,
    n_train: int = 40,
    n_val: int = 12,
    n_test: int = 12,
) -> tuple[Path, dict, Any]:
    if model_type not in {"forest", "logreg"}:
        raise ValueError("Model type must be forest or logreg.")
    dest = resolve_under_checkpoints(
        output_name,
        checkpoints_dir_from_config(config),
        default_name="activity_custom.joblib",
    )
    if dest.name == DEMO_CHECKPOINT_NAME and dest.exists() and not confirm_overwrite_demo:
        raise ValueError(
            "Overwriting the bundled demo checkpoint requires confirmation."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle, metrics = train_activity_model(
            data_root=data_root_from_config(config),
            output=dest,
            model_type=model_type,
            seed=seed,
            generate_demo=generate_demo,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            overwrite_demo=overwrite_demo_images,
        )
    except SystemExit as exc:
        raise ValueError(str(exc) or "Training failed.") from exc
    return dest, metrics, bundle


def run_evaluation(config: dict, split: str = "test", checkpoint: str | Path | None = None) -> dict:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    ckpt = Path(checkpoint) if checkpoint is not None else Path(config["ACTIVITY_CHECKPOINT"])
    try:
        return evaluate_checkpoint(ckpt, data_root_from_config(config), split)
    except SystemExit as exc:
        raise ValueError(str(exc) or "Evaluation failed.") from exc


def metrics_as_text(metrics: dict) -> dict[str, str]:
    reports = {}
    for split in ("val", "test"):
        if isinstance(metrics.get(split), dict) and "accuracy" in metrics[split]:
            reports[split] = format_metrics_report(metrics[split])
    return reports


def _safe_zip_parts(name: str) -> list[str] | None:
    if not name or name.endswith("/"):
        return None
    if name.startswith("/") or name.startswith("\\"):
        return None
    raw = name.replace("\\", "/")
    parts = [p for p in Path(raw).parts if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        return None
    if parts[0] == "__MACOSX" or parts[-1] in {".DS_Store", "Thumbs.db"}:
        return None
    return parts


def labeled_dest_folder(root: Path, split: str, folder_name: str) -> Path:
    """Resolve a category or sport folder onto the on-disk dataset path."""
    category, sport = parse_folder_label(folder_name)
    if sport:
        return root / split / f"game_or_play__{sport}"
    return root / split / category


def folder_label_for_upload(category: str, sport_context: str = "") -> str:
    sport = (sport_context or "").strip()
    if sport:
        resolved = resolve_sport(sport)
        if resolved is None:
            raise ValueError(f"Unknown sport context '{sport}'.")
        return f"game_or_play__{resolved.id}"
    parse_folder_label(category)
    return category


def _classify_zip_member(
    parts: list[str], default_split: str, default_category: str | None
) -> tuple[str, str] | None:
    if len(parts) >= 3 and parts[-3] in SPLITS:
        try:
            parse_folder_label(parts[-2])
            return parts[-3], parts[-2]
        except ValueError:
            return None
    if len(parts) >= 2:
        try:
            parse_folder_label(parts[-2])
            return default_split, parts[-2]
        except ValueError:
            if default_category:
                parse_folder_label(default_category)
                return default_split, default_category
            return None
    if default_category:
        parse_folder_label(default_category)
        return default_split, default_category
    return None


def save_labeled_files(
    config: dict,
    files: Iterable,
    *,
    category: str,
    split: str = "train",
    sport_context: str = "",
) -> int:
    split = split.strip().lower() or "train"
    if split not in SPLITS:
        raise ValueError("Split must be train, val, or test.")
    folder = folder_label_for_upload(category, sport_context)
    root = ensure_dataset_tree(data_root_from_config(config))
    dest_dir = labeled_dest_folder(root, split, folder)
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for upload in files:
        if upload is None or not getattr(upload, "filename", None):
            continue
        name = Path(upload.filename).name
        suffix = Path(name).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {name}")
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
        if not safe or safe.startswith("."):
            raise ValueError(f"Invalid filename: {name}")
        dest = dest_dir / safe
        upload.save(dest)
        if dest.stat().st_size > MAX_UPLOAD_BYTES:
            dest.unlink(missing_ok=True)
            raise ValueError(f"{name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        saved += 1
    if saved == 0:
        raise ValueError("No image files were uploaded.")
    return saved


def save_labeled_zip(
    config: dict,
    upload,
    *,
    default_category: str | None = None,
    default_split: str = "train",
) -> int:
    if upload is None or not getattr(upload, "filename", None):
        raise ValueError("Choose a zip of train/<category>/*.jpg.")
    payload = upload.read()
    if not payload:
        raise ValueError("Zip file is empty.")
    if len(payload) > MAX_UPLOAD_BYTES * 4:
        raise ValueError("Zip file is too large.")
    try:
        archive = zipfile.ZipFile(io.BytesIO(payload))
    except zipfile.BadZipFile as exc:
        raise ValueError("File is not a valid zip archive.") from exc

    default_split = default_split.strip().lower() or "train"
    if default_split not in SPLITS:
        raise ValueError("Split must be train, val, or test.")
    if default_category:
        parse_folder_label(default_category)

    root = ensure_dataset_tree(data_root_from_config(config))
    names = archive.namelist()
    if len(names) > MAX_ZIP_MEMBERS:
        raise ValueError(f"Zip has too many members (max {MAX_ZIP_MEMBERS}).")

    saved = 0
    for name in names:
        parts = _safe_zip_parts(name)
        if parts is None:
            if name.endswith("/") or name.startswith("__MACOSX"):
                continue
            if ".." in name.replace("\\", "/") or name.startswith("/") or name.startswith("\\"):
                raise ValueError(f"Rejected path traversal in zip member: {name}")
            continue
        classified = _classify_zip_member(parts, default_split, default_category)
        if classified is None:
            continue
        split, category = classified
        suffix = Path(parts[-1]).suffix.lower()
        if suffix not in IMAGE_SUFFIXES:
            continue
        info = archive.getinfo(name)
        if info.file_size > MAX_UPLOAD_BYTES:
            raise ValueError(f"{parts[-1]} exceeds the per-file size limit.")
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in parts[-1])
        dest = labeled_dest_folder(root, split, category) / safe
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(archive.read(name))
        saved += 1
    if saved == 0:
        raise ValueError(
            "Zip contained no recognized labeled frames. "
            f"Use train/<category>/*.jpg with categories: {', '.join(ACTIVITY_CATEGORIES)}."
        )
    return saved
