"""Classification metrics for Phase 3 activity recognition.

Accuracy plus per-class precision / recall / F1. Implemented without
requiring scikit-learn at import time so tests and reports stay lightweight.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    labels: Sequence[str],
) -> dict:
    """Return accuracy, per-class P/R/F1, macro averages, and a confusion matrix."""
    truth = list(y_true)
    pred = list(y_pred)
    if len(truth) != len(pred):
        raise ValueError("y_true and y_pred length mismatch")

    n = len(truth)
    accuracy = (sum(t == p for t, p in zip(truth, pred)) / n) if n else 0.0

    per_class: dict[str, dict] = {}
    for lab in labels:
        tp = sum(t == lab and p == lab for t, p in zip(truth, pred))
        fp = sum(t != lab and p == lab for t, p in zip(truth, pred))
        fn = sum(t == lab and p != lab for t, p in zip(truth, pred))
        support = sum(t == lab for t in truth)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall)
            else 0.0
        )
        per_class[lab] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": int(support),
        }

    if labels:
        macro = {
            "precision": round(
                sum(per_class[l]["precision"] for l in labels) / len(labels), 4
            ),
            "recall": round(
                sum(per_class[l]["recall"] for l in labels) / len(labels), 4
            ),
            "f1": round(sum(per_class[l]["f1"] for l in labels) / len(labels), 4),
        }
    else:
        macro = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    matrix = [[0 for _ in labels] for _ in labels]
    index = {lab: i for i, lab in enumerate(labels)}
    for t, p in zip(truth, pred):
        if t in index and p in index:
            matrix[index[t]][index[p]] += 1

    return {
        "accuracy": round(accuracy, 4),
        "n_samples": n,
        "per_class": per_class,
        "macro": macro,
        "labels": list(labels),
        "confusion_matrix": matrix,
    }


def format_metrics_report(metrics: Mapping) -> str:
    """Human-readable metrics table for CLI / docs."""
    labels: Iterable[str] = metrics.get("labels") or metrics.get("per_class", {}).keys()
    lines = [
        f"accuracy: {metrics.get('accuracy', 0):.4f}  (n={metrics.get('n_samples', 0)})",
        f"macro  P={metrics['macro']['precision']:.4f}  "
        f"R={metrics['macro']['recall']:.4f}  F1={metrics['macro']['f1']:.4f}",
        "",
        f"{'class':<28} {'prec':>7} {'rec':>7} {'f1':>7} {'sup':>5}",
        "-" * 56,
    ]
    per_class = metrics.get("per_class", {})
    for lab in labels:
        row = per_class.get(lab, {})
        lines.append(
            f"{lab:<28} {row.get('precision', 0):7.3f} {row.get('recall', 0):7.3f} "
            f"{row.get('f1', 0):7.3f} {row.get('support', 0):5d}"
        )
    labels_list = list(labels)
    matrix = metrics.get("confusion_matrix")
    if matrix and labels_list:
        lines.append("")
        lines.append("confusion matrix (rows=true, cols=pred):")
        header = " ".join(f"{i:>3}" for i in range(len(labels_list)))
        lines.append(f"     {header}")
        for i, lab in enumerate(labels_list):
            cells = " ".join(f"{v:>3}" for v in matrix[i])
            lines.append(f"[{i}] {cells}   {lab}")
    return "\n".join(lines)
