"""Optional CPU sklearn trainer for labeled object frames.

This does **not** replace YOLO inventory. It fits a small whole-frame
classifier when operators have labeled crops/frames under
``data/objects/{train,val,test}/<object_id>/``. Runtime detection still
prefers YOLO; this checkpoint is for admin iteration only and is not
used to invent refrigerators on real uploads.

    python -m vision.train_objects --data-root data/objects --output data/checkpoints/objects_custom.joblib
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np

from vision.dataset import IMAGE_SUFFIXES
from vision.features import FEATURE_DIM, FEATURE_VERSION, extract_frame_features
from vision.metrics import compute_classification_metrics, format_metrics_report
from vision.objects_catalog import (
    MIN_OBJECT_TRAIN_CLASSES,
    MIN_OBJECT_TRAIN_IMAGES,
    describe_objects_dataset,
    resolve_object,
)

logger = logging.getLogger(__name__)


def _sklearn():
    try:
        import sklearn
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise SystemExit(
            "scikit-learn is required to train. pip install -r requirements.txt"
        ) from exc
    return sklearn, RandomForestClassifier, LogisticRegression, StandardScaler


def build_estimator(model_type: str, seed: int):
    _, RandomForestClassifier, LogisticRegression, _ = _sklearn()
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=seed,
        )
    if model_type != "forest":
        raise ValueError(f"Unknown model type: {model_type}")
    return RandomForestClassifier(
        n_estimators=60,
        max_depth=10,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def _load_split(root: Path, split: str) -> tuple[np.ndarray, list[str]]:
    split_dir = root / split
    features: list[np.ndarray] = []
    labels: list[str] = []
    if not split_dir.is_dir():
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), []
    for child in sorted(split_dir.iterdir()):
        if not child.is_dir():
            continue
        entry = resolve_object(child.name)
        if entry is None:
            continue
        for path in sorted(child.iterdir()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            features.append(extract_frame_features(image))
            labels.append(entry.id)
    if not features:
        return np.zeros((0, FEATURE_DIM), dtype=np.float32), []
    return np.stack(features).astype(np.float32), labels


def train_object_model(
    data_root: str | Path,
    output: str | Path,
    model_type: str = "forest",
    seed: int = 7,
) -> tuple[dict[str, Any], dict]:
    """Fit a small sklearn classifier. Fails closed if too little data."""
    sklearn, *_ = _sklearn()
    _, _, _, StandardScaler = _sklearn()

    root = Path(data_root)
    inventory = describe_objects_dataset(root)
    labeled = inventory.get("labeled") or {}
    enough = {
        oid: n for oid, n in labeled.items() if int(n) >= MIN_OBJECT_TRAIN_IMAGES
    }
    if len(enough) < MIN_OBJECT_TRAIN_CLASSES:
        raise SystemExit(
            "Not enough labeled object frames to train. Need at least "
            f"{MIN_OBJECT_TRAIN_CLASSES} catalog classes with "
            f"{MIN_OBJECT_TRAIN_IMAGES}+ images each under {root}/train/. "
            "Extract frames from authorized video or upload crops. "
            "This trainer does not invent detections."
        )

    X_train, y_train = _load_split(root, "train")
    if len(y_train) == 0:
        raise SystemExit(f"No training images under {root}/train/<object_id>/. ")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = build_estimator(model_type, seed)
    clf.fit(Xs, y_train)
    classes = sorted(set(y_train))

    metrics: dict = {"dataset": inventory}
    for split in ("val", "test"):
        X, y = _load_split(root, split)
        if len(y) == 0:
            continue
        pred = clf.predict(scaler.transform(X))
        metrics[split] = compute_classification_metrics(y, pred.tolist(), classes)

    bundle = {
        "model": clf,
        "scaler": scaler,
        "categories": classes,
        "feature_version": FEATURE_VERSION,
        "feature_dim": FEATURE_DIM,
        "metrics": metrics,
        "sklearn_version": getattr(sklearn, "__version__", ""),
        "model_type": model_type,
        "backend": "opencv_sklearn_objects",
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "notes": (
            "Optional whole-frame object classifier (OpenCV features + sklearn). "
            "Assistive only. Runtime inventory prefers YOLO and will not invent "
            "objects when the detector is missing."
        ),
    }
    dest = Path(output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, dest)
    logger.info("Wrote object checkpoint %s classes=%s", dest, classes)
    return bundle, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train optional object classifier")
    parser.add_argument("--data-root", default="data/objects")
    parser.add_argument("--output", default="data/checkpoints/objects_custom.joblib")
    parser.add_argument("--model", choices=("forest", "logreg"), default="forest")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        _bundle, metrics = train_object_model(
            args.data_root,
            args.output,
            model_type=args.model,
            seed=args.seed,
        )
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        return 1
    for split in ("val", "test"):
        if split in metrics:
            print(format_metrics_report(metrics[split]))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
