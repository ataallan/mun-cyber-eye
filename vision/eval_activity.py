"""Evaluate a Phase 3 activity checkpoint (accuracy / precision / recall / F1).

Examples::

    python -m vision.eval_activity --split test
    python -m vision.eval_activity --checkpoint data/checkpoints/activity_demo.joblib --json report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from vision.activity import load_checkpoint
from vision.dataset import ACTIVITY_CATEGORIES, load_split_features
from vision.metrics import compute_classification_metrics, format_metrics_report


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_root: str | Path,
    split: str = "test",
) -> dict:
    bundle = load_checkpoint(checkpoint)
    X, y, paths = load_split_features(data_root, split)
    if len(y) == 0:
        raise SystemExit(
            f"No images in {data_root}/{split}/<category>/. "
            "Generate a demo set or point --data-root at labeled frames."
        )
    Xs = bundle.scaler.transform(X) if bundle.scaler is not None else X
    pred = bundle.model.predict(Xs)
    metrics = compute_classification_metrics(
        y.tolist(), pred.tolist(), ACTIVITY_CATEGORIES
    )
    metrics["checkpoint"] = str(checkpoint)
    metrics["split"] = split
    metrics["n_paths"] = len(paths)
    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        description="Evaluate Mun Cyber Eye Phase 3 activity recognition."
    )
    p.add_argument(
        "--data-root",
        default=str(root / "data" / "activity"),
    )
    p.add_argument(
        "--checkpoint",
        default=str(root / "data" / "checkpoints" / "activity_demo.joblib"),
    )
    p.add_argument("--split", choices=("train", "val", "test"), default="test")
    p.add_argument("--json", dest="json_out", default="", help="Optional JSON report path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    metrics = evaluate_checkpoint(args.checkpoint, args.data_root, args.split)
    print(format_metrics_report(metrics))
    if args.json_out:
        dest = Path(args.json_out)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
