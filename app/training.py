"""Helpers for the developer (Mun Cyber) Train models console.

Thin wrappers around ``vision.train_activity`` / ``vision.eval_activity`` plus
safe upload and active-checkpoint activation. Training math stays in vision/.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Iterable

import cv2

from ingest.sampler import FrameSampler
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
    parse_folder_tags,
)
from vision.dangerous_objects import (
    describe_dangerous_dataset,
    ensure_dangerous_tree,
    validate_dangerous_id,
)
from vision.objects_catalog import (
    describe_objects_dataset,
    ensure_objects_tree,
    validate_object_id,
)
from vision.scene_context import validate_place_type
from vision.sports_catalog import resolve_sport
from vision.eval_activity import evaluate_checkpoint
from vision.metrics import format_metrics_report
from vision.train_activity import train_activity_model

DEMO_CHECKPOINT_NAME = "activity_demo.joblib"
OBJECT_CHECKPOINT_DEFAULT = "objects_custom.joblib"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_VIDEO_UPLOAD_BYTES = 64 * 1024 * 1024
MAX_ZIP_MEMBERS = 400
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
CLIP_VIDEO_SUFFIXES = VIDEO_SUFFIXES | {".webm"}
VIDEO_EXTRACT_KINDS = ("activity", "sport", "place", "object", "dangerous")
MAX_VIDEO_EXTRACT_FRAMES = 240
NO_CLIP_MESSAGE = "No clip to learn from."


def data_root_from_config(config: dict) -> Path:
    return Path(config["ACTIVITY_DATA_ROOT"])


def objects_root_from_config(config: dict) -> Path:
    return Path(config["OBJECTS_DATA_ROOT"])


def checkpoints_dir_from_config(config: dict) -> Path:
    return Path(config["CHECKPOINTS_DIR"])


def active_file_from_config(config: dict) -> Path:
    return Path(config["ACTIVE_CHECKPOINT_FILE"])


def dataset_inventory(config: dict) -> dict:
    root = data_root_from_config(config)
    ensure_dataset_tree(root)
    return describe_dataset(root)


def objects_inventory(config: dict) -> dict:
    root = objects_root_from_config(config)
    root.mkdir(parents=True, exist_ok=True)
    return describe_objects_dataset(root)


def dangerous_root_from_config(config: dict) -> Path:
    return Path(
        config.get("DANGEROUS_DATA_ROOT")
        or Path(config["OBJECTS_DATA_ROOT"]).parent / "dangerous"
    )


def dangerous_inventory(config: dict) -> dict:
    root = dangerous_root_from_config(config)
    root.mkdir(parents=True, exist_ok=True)
    return describe_dangerous_dataset(root)


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
            "Overwriting the bundled checkpoint requires confirmation."
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
    """Resolve a category, sport, or scene folder onto the on-disk path."""
    tags = parse_folder_tags(folder_name)
    if tags.sport_context:
        return root / split / f"game_or_play__{tags.sport_context}"
    if tags.place_type:
        prefix = folder_name.strip().lower()
        if prefix.startswith("scene__") or "__scene__" in prefix:
            if tags.category not in {"ordinary", "game_or_play"} and "scene__" in prefix:
                return root / split / f"{tags.category}__scene__{tags.place_type}"
            return root / split / f"scene__{tags.place_type}"
        return root / split / f"scene__{tags.place_type}"
    return root / split / tags.category


def folder_label_for_upload(
    category: str, sport_context: str = "", place_type: str = ""
) -> str:
    sport = (sport_context or "").strip()
    place = (place_type or "").strip()
    if sport:
        resolved = resolve_sport(sport)
        if resolved is None:
            raise ValueError(f"Unknown sport context '{sport}'.")
        return f"game_or_play__{resolved.id}"
    if place:
        place_id = validate_place_type(place)
        cat = (category or "").strip()
        if cat and cat not in {"", "ordinary", "game_or_play"}:
            parse_folder_label(cat)
            return f"{cat}__scene__{place_id}"
        return f"scene__{place_id}"
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
    place_type: str = "",
) -> int:
    split = split.strip().lower() or "train"
    if split not in SPLITS:
        raise ValueError("Split must be train, val, or test.")
    folder = folder_label_for_upload(category, sport_context, place_type)
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
            f"Use train/<category>/*.jpg with categories: {', '.join(ACTIVITY_CATEGORIES)} "
            "or scene__<place_type> / catalog sport folders."
        )
    return saved


def _safe_upload_name(name: str) -> str:
    base = Path(name or "").name
    if not base or base in {".", ".."}:
        raise ValueError("Invalid filename.")
    if ".." in base.replace("\\", "/"):
        raise ValueError(f"Rejected path traversal: {name}")
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in base)
    if not safe or safe.startswith("."):
        raise ValueError(f"Invalid filename: {name}")
    return safe


def _assert_under(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    root_res = root.resolve()
    if resolved != root_res and root_res not in resolved.parents:
        raise ValueError("Rejected path traversal outside the dataset directory.")
    return resolved


def resolve_label_dest(
    config: dict,
    *,
    kind: str,
    split: str,
    category: str = "",
    sport_context: str = "",
    place_type: str = "",
    object_id: str = "",
) -> tuple[Path, str]:
    """Return (directory, folder_label) for extracted or uploaded labels."""
    split = (split or "train").strip().lower() or "train"
    if split not in SPLITS:
        raise ValueError("Split must be train, val, or test.")
    kind = (kind or "activity").strip().lower()
    if kind not in VIDEO_EXTRACT_KINDS:
        raise ValueError(
            "Kind must be activity, sport, place, object, or dangerous (catalog class)."
        )
    if kind == "object":
        oid = validate_object_id(object_id)
        root = ensure_objects_tree(objects_root_from_config(config), [oid])
        dest = _assert_under(root / split / oid, root)
        dest.mkdir(parents=True, exist_ok=True)
        return dest, f"object:{oid}"
    if kind == "dangerous":
        oid = validate_dangerous_id(object_id)
        root = ensure_dangerous_tree(dangerous_root_from_config(config), [oid])
        dest = _assert_under(root / split / oid, root)
        dest.mkdir(parents=True, exist_ok=True)
        return dest, f"dangerous:{oid}"

    if kind == "sport":
        folder = folder_label_for_upload("game_or_play", sport_context, "")
    elif kind == "place":
        folder = folder_label_for_upload(category or "ordinary", "", place_type)
    else:
        folder = folder_label_for_upload(category, "", "")
    root = ensure_dataset_tree(data_root_from_config(config))
    dest = labeled_dest_folder(root, split, folder)
    dest = _assert_under(dest, root)
    dest.mkdir(parents=True, exist_ok=True)
    return dest, folder


def _clamp_extract_limits(sample_fps: float, max_frames: int) -> tuple[float, int]:
    try:
        fps = float(sample_fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("Sample FPS must be a number.") from exc
    fps = max(0.1, min(fps, 15.0))
    try:
        cap = int(max_frames)
    except (TypeError, ValueError) as exc:
        raise ValueError("Max frames must be an integer.") from exc
    cap = max(1, min(cap, MAX_VIDEO_EXTRACT_FRAMES))
    return fps, cap


def _safe_frame_stem(stem: str, fallback: str = "frame") -> str:
    raw = (stem or "").strip() or fallback
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in raw)
    safe = safe.strip("._")[:40]
    return safe or fallback


def _write_sampled_frames(
    video_path: Path,
    dest_dir: Path,
    *,
    sample_fps: float,
    max_frames: int,
    stem: str,
    source_label: str,
) -> int:
    """Sample ``video_path`` with FrameSampler and write JPEGs into ``dest_dir``."""
    sampler = FrameSampler(
        video_path,
        sample_fps=sample_fps,
        max_frames=max_frames,
        source_label=source_label,
    )
    written = 0
    safe_stem = _safe_frame_stem(stem)
    for frame in sampler.frames():
        name = f"{safe_stem}_f{frame.index:05d}.jpg"
        out = dest_dir / name
        _assert_under(out, dest_dir)
        ok = cv2.imwrite(str(out), frame.image_bgr)
        if not ok:
            raise ValueError(f"Failed to write sampled frame {name}.")
        written += 1
    return written


def correction_label_args(
    category: str, sport_context: str = "", place_type: str = ""
) -> dict[str, str]:
    """Map a review correction onto the existing activity / sport / place folders."""
    sport = (sport_context or "").strip()
    place = (place_type or "").strip()
    cat = (category or "").strip()
    if sport and place:
        raise ValueError("Choose a sport context or a place.")
    if sport:
        return {
            "kind": "sport",
            "category": "game_or_play",
            "sport_context": sport,
            "place_type": "",
        }
    if place:
        return {
            "kind": "place",
            "category": cat or "ordinary",
            "sport_context": "",
            "place_type": place,
        }
    if not cat:
        raise ValueError("Choose an activity class.")
    return {
        "kind": "activity",
        "category": cat,
        "sport_context": "",
        "place_type": "",
    }


def _contained_video(path: Path, root: Path) -> Path | None:
    try:
        root_res = root.resolve()
        resolved = path.resolve()
    except OSError:
        return None
    if not resolved.is_file():
        return None
    if resolved.suffix.lower() not in CLIP_VIDEO_SUFFIXES:
        return None
    if resolved != root_res and root_res not in resolved.parents:
        return None
    return resolved


def _clip_path_candidates(alert: Any) -> list[str]:
    paths: list[str] = []
    clip = str(getattr(alert, "clip_path", None) or "").strip()
    if clip:
        paths.append(clip)
    raw = getattr(alert, "metadata_json", None) or ""
    meta: Any = {}
    if raw:
        try:
            meta = json.loads(raw)
        except json.JSONDecodeError:
            meta = {}
    incident = meta.get("incident_clip") if isinstance(meta, dict) else None
    if isinstance(incident, dict):
        extra = str(incident.get("path") or "").strip()
        if extra and extra not in paths:
            paths.append(extra)
    return paths


def learning_video_for_alert(
    config: dict,
    alert: Any,
    *,
    archive_root: str | Path | None = None,
    segments: Iterable[Any] | None = None,
) -> dict[str, Any]:
    """Prefer the incident clip. Fall back to an archive segment that covers the alert.

    Raises ValueError with ``No clip to learn from.`` when neither file is usable.
    Paths outside the clip or archive directory are ignored.
    """
    clip_root = Path(config.get("CLIP_DIR") or "")
    for raw in _clip_path_candidates(alert):
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = clip_root / candidate
        found = _contained_video(candidate, clip_root)
        if found is not None:
            return {"path": found, "kind": "incident_clip", "name": found.name}

    camera_id = str(getattr(alert, "camera_id", "") or "").strip()
    when = str(getattr(alert, "created_at", "") or "").strip()
    root = Path(archive_root) if archive_root is not None else Path(config.get("ARCHIVE_DIR") or "")
    if camera_id and when and segments:
        matches = []
        for seg in segments:
            if str(getattr(seg, "camera_id", "") or "") != camera_id:
                continue
            started = str(getattr(seg, "started_at", "") or "")
            ended = str(getattr(seg, "ended_at", "") or "")
            if started and when < started:
                continue
            if ended and when > ended:
                continue
            if not started and not ended:
                continue
            rel_raw = str(getattr(seg, "relpath", "") or "").strip()
            if not rel_raw:
                continue
            rel = Path(rel_raw)
            if rel.is_absolute() or any(part == ".." for part in rel.parts):
                continue
            found = _contained_video(root / rel, root)
            if found is None:
                continue
            matches.append((started, found, str(getattr(seg, "id", "") or "")))
        matches.sort(key=lambda item: item[0], reverse=True)
        if matches:
            _started, found, segment_id = matches[0]
            return {
                "path": found,
                "kind": "archive_segment",
                "name": found.name,
                "segment_id": segment_id,
            }
    raise ValueError(NO_CLIP_MESSAGE)


def extract_clip_frames(
    config: dict,
    video_path: str | Path,
    *,
    kind: str,
    split: str = "train",
    category: str = "",
    sport_context: str = "",
    place_type: str = "",
    object_id: str = "",
    sample_fps: float = 2.0,
    max_frames: int = 40,
    stem: str = "clip",
    source_label: str = "",
) -> dict[str, Any]:
    """Sample an on-disk clip into labeled JPEGs. Does not train or activate."""
    path = Path(video_path)
    if not path.is_file():
        raise ValueError(NO_CLIP_MESSAGE)
    if path.suffix.lower() not in CLIP_VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video type: {path.name}. Use mp4, avi, mov, mkv, or webm."
        )
    fps, cap = _clamp_extract_limits(sample_fps, max_frames)
    dest_dir, folder = resolve_label_dest(
        config,
        kind=kind,
        split=split,
        category=category,
        sport_context=sport_context,
        place_type=place_type,
        object_id=object_id,
    )
    label = source_label or f"authorized clip — {path.name}"
    try:
        written = _write_sampled_frames(
            path,
            dest_dir,
            sample_fps=fps,
            max_frames=cap,
            stem=stem,
            source_label=label,
        )
    except FileNotFoundError as exc:
        raise ValueError(NO_CLIP_MESSAGE) from exc
    except RuntimeError as exc:
        raise ValueError(str(exc) or "Unable to open the video.") from exc
    if written == 0:
        raise ValueError(
            "No frames could be sampled from that clip. "
            "Check the file is a valid incident clip."
        )
    return {
        "frames": written,
        "folder": folder,
        "dest": str(dest_dir),
        "split": split,
        "kind": kind,
        "sample_fps": fps,
        "max_frames": cap,
        "source": path.name,
    }


def save_gallery_files(
    config: dict,
    files: Iterable,
    *,
    kind: str,
    object_id: str,
    split: str = "train",
) -> int:
    """Save uploaded examples under data/objects or data/dangerous."""
    dest_dir, _folder = resolve_label_dest(
        config,
        kind=kind,
        split=split,
        object_id=object_id,
    )
    root = dest_dir.parent.parent
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
        dest = _assert_under(dest_dir / safe, root)
        upload.save(dest)
        if dest.stat().st_size > MAX_UPLOAD_BYTES:
            dest.unlink(missing_ok=True)
            raise ValueError(f"{name} exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
        saved += 1
    if saved == 0:
        raise ValueError("No image files were uploaded.")
    return saved


def extract_video_frames(
    config: dict,
    upload,
    *,
    kind: str,
    split: str = "train",
    category: str = "",
    sport_context: str = "",
    place_type: str = "",
    object_id: str = "",
    sample_fps: float = 2.0,
    max_frames: int = 60,
) -> dict[str, Any]:
    """Sample an authorized video into labeled JPEG frames via FrameSampler.

    Videos are not trained on directly — the sklearn activity trainer still
    learns from images. Frames land under data/activity/... or
    data/objects/train/<object_id>/.
    """
    if upload is None or not getattr(upload, "filename", None):
        raise ValueError("Choose an authorized mp4 / avi / mov / mkv video.")
    raw_name = upload.filename or ""
    safe = _safe_upload_name(raw_name)
    suffix = Path(safe).suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise ValueError(
            f"Unsupported video type: {safe}. Use mp4, avi, mov, or mkv."
        )
    payload = upload.read()
    if not payload:
        raise ValueError("Video file is empty.")
    if len(payload) > MAX_VIDEO_UPLOAD_BYTES:
        raise ValueError(
            f"Video exceeds the {MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)} MB limit."
        )

    fps, cap = _clamp_extract_limits(sample_fps, max_frames)

    dest_dir, folder = resolve_label_dest(
        config,
        kind=kind,
        split=split,
        category=category,
        sport_context=sport_context,
        place_type=place_type,
        object_id=object_id,
    )

    upload_dir = Path(config.get("UPLOAD_DIR") or dest_dir.parent)
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / safe
    video_path.write_bytes(payload)

    try:
        written = _write_sampled_frames(
            video_path,
            dest_dir,
            sample_fps=fps,
            max_frames=cap,
            stem=Path(safe).stem,
            source_label=f"authorized labeling video — {safe}",
        )
    except FileNotFoundError as exc:
        raise ValueError(str(exc)) from exc
    except RuntimeError as exc:
        raise ValueError(str(exc) or "Unable to open the video.") from exc

    if written == 0:
        raise ValueError(
            "No frames could be sampled from that video. "
            "Check the file is a valid authorized clip."
        )
    return {
        "frames": written,
        "folder": folder,
        "dest": str(dest_dir),
        "split": split,
        "kind": kind,
        "sample_fps": fps,
        "max_frames": cap,
        "source": safe,
    }


def run_object_training(
    config: dict,
    *,
    model_type: str = "forest",
    output_name: str = OBJECT_CHECKPOINT_DEFAULT,
    seed: int = 7,
) -> tuple[Path, dict, dict]:
    from vision.train_objects import train_object_model

    if model_type not in {"forest", "logreg"}:
        raise ValueError("Model type must be forest or logreg.")
    dest = resolve_under_checkpoints(
        output_name,
        checkpoints_dir_from_config(config),
        default_name=OBJECT_CHECKPOINT_DEFAULT,
    )
    if dest.name == DEMO_CHECKPOINT_NAME:
        raise ValueError("Object training must not overwrite the activity demo checkpoint.")
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        bundle, metrics = train_object_model(
            data_root=objects_root_from_config(config),
            output=dest,
            model_type=model_type,
            seed=seed,
        )
    except SystemExit as exc:
        raise ValueError(str(exc) or "Object training failed.") from exc
    return dest, metrics, bundle
