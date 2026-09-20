"""Train the Phase 3 activity-recognition model (CPU / sklearn).

Examples::

    python -m vision.train_activity --generate-demo --output data/checkpoints/activity_demo.joblib
    python -m vision.train_activity --data-root data/activity --model forest
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from vision.activity import ActivityCheckpoint, save_checkpoint
from vision.dataset import (
    ACTIVITY_CATEGORIES,
    describe_dataset,
    generate_demo_dataset,
    load_split_features,
)
from vision.features import FEATURE_DIM, FEATURE_VERSION
from vision.metrics import compute_classification_metrics, format_metrics_report

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
        n_estimators=80,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=1,
    )


def train_activity_model(
    data_root: str | Path,
    output: str | Path,
    model_type: str = "forest",
    seed: int = 7,
    generate_demo: bool = False,
    n_train: int = 40,
    n_val: int = 12,
    n_test: int = 12,
    overwrite_demo: bool = False,
) -> tuple[ActivityCheckpoint, dict]:
    """Fit scaler + classifier, evaluate val/test if present, write checkpoint."""
    sklearn, *_ = _sklearn()
    _, _, _, StandardScaler = _sklearn()

    data_root = Path(data_root)
    if generate_demo:
        generate_demo_dataset(
            data_root,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=seed,
            overwrite=overwrite_demo,
        )

    X_train, y_train, _ = load_split_features(data_root, "train")
    if len(y_train) == 0:
        raise SystemExit(
            f"No training images under {data_root}/train/<category>/. "
            "Add labeled frames or pass --generate-demo."
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    clf = build_estimator(model_type, seed)
    clf.fit(Xs, y_train)

    metrics: dict = {"dataset": describe_dataset(data_root)}
    for split in ("val", "test"):
        X, y, _ = load_split_features(data_root, split)
        if len(y) == 0:
            continue
        pred = clf.predict(scaler.transform(X))
        metrics[split] = compute_classification_metrics(
            y.tolist(), pred.tolist(), ACTIVITY_CATEGORIES
        )

    bundle = ActivityCheckpoint(
        model=clf,
        scaler=scaler,
        categories=list(ACTIVITY_CATEGORIES),
        feature_version=FEATURE_VERSION,
        feature_dim=FEATURE_DIM,
        metrics=metrics,
        sklearn_version=getattr(sklearn, "__version__", ""),
        model_type=model_type,
    )
    save_checkpoint(bundle, output)
    return bundle, metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Train Mun Cyber Eye Phase 3 activity recognition (CPU)."
    )
    p.add_argument(
        "--data-root",
        default=str(root / "data" / "activity"),
        help="Dataset root (train/val/test/<category>/)",
    )
    p.add_argument(
        "--output",
        default=str(root / "data" / "checkpoints" / "activity_demo.joblib"),
        help="Checkpoint path (default: data/checkpoints/activity_demo.joblib)",
    )
    p.add_argument(
        "--model",
        dest="model_type",
        choices=("forest", "logreg"),
        default="forest",
        help="sklearn estimator (RandomForest is the demo default)",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--generate-demo",
        action="store_true",
        help="Synthesize geometric demo images if the layout is empty",
    )
    p.add_argument("--overwrite-demo", action="store_true")
    p.add_argument("--n-train", type=int, default=40)
    p.add_argument("--n-val", type=int, default=12)
    p.add_argument("--n-test", type=int, default=12)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    bundle, metrics = train_activity_model(
        data_root=args.data_root,
        output=args.output,
        model_type=args.model_type,
        seed=args.seed,
        generate_demo=args.generate_demo,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        overwrite_demo=args.overwrite_demo,
    )
    print(f"checkpoint: {args.output}")
    print(f"model: {bundle.model_type}  sklearn={bundle.sklearn_version}")
    print(f"feature_version={bundle.feature_version} dim={bundle.feature_dim}")
    for split in ("val", "test"):
        if split in metrics:
            print()
            print(f"=== {split} ===")
            print(format_metrics_report(metrics[split]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
