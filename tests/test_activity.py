"""Phase 3 activity recognition: features, metrics, train/eval, adapter fallback."""

import numpy as np
import pytest

from vision.dataset import (
    ACTIVITY_CATEGORIES,
    generate_demo_dataset,
    render_demo_frame,
)
from vision.detector import create_adapter
from vision.features import FEATURE_DIM, extract_frame_features, extract_stack_features
from vision.metrics import compute_classification_metrics, format_metrics_report
from vision.train_activity import train_activity_model


def test_feature_vector_is_finite_and_fixed():
    img = render_demo_frame("ordinary", seed=1)
    vec = extract_frame_features(img)
    assert vec.shape == (FEATURE_DIM,)
    assert vec.dtype == np.float32
    assert np.isfinite(vec).all()


def test_stack_features_use_motion():
    a = render_demo_frame("ordinary", seed=2)
    b = render_demo_frame("potential_fight", seed=3)
    still = extract_stack_features([a])
    moved = extract_stack_features([a, b])
    assert still.shape == moved.shape == (FEATURE_DIM,)
    # last 4 stats include motion; they should differ when a previous frame exists
    assert still.shape[0] == FEATURE_DIM


def test_metrics_per_class_keys():
    y_true = ["ordinary", "potential_fight", "ordinary", "potential_fall"]
    y_pred = ["ordinary", "potential_fight", "potential_fight", "potential_fall"]
    metrics = compute_classification_metrics(y_true, y_pred, ACTIVITY_CATEGORIES)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] == pytest.approx(0.75)
    for cat in ACTIVITY_CATEGORIES:
        row = metrics["per_class"][cat]
        assert {"precision", "recall", "f1", "support"} <= set(row)
    report = format_metrics_report(metrics)
    assert "potential_fight" in report
    assert "confusion matrix" in report


def _train_tiny(tmp_path):
    data_root = tmp_path / "activity"
    ckpt = tmp_path / "activity_demo.joblib"
    bundle, metrics = train_activity_model(
        data_root=data_root,
        output=ckpt,
        model_type="forest",
        seed=7,
        generate_demo=True,
        n_train=8,
        n_val=4,
        n_test=4,
        overwrite_demo=True,
    )
    return ckpt, bundle, metrics


def test_train_and_eval_synthetic(tmp_path):
    ckpt, bundle, metrics = _train_tiny(tmp_path)
    assert ckpt.is_file()
    assert bundle.feature_dim == FEATURE_DIM
    assert "test" in metrics
    # Synthetic classes are visually distinct; a tiny forest should separate them.
    assert metrics["test"]["accuracy"] >= 0.75
    for cat in ACTIVITY_CATEGORIES:
        assert cat in metrics["test"]["per_class"]
        assert "f1" in metrics["test"]["per_class"][cat]


def test_adapter_predicts_demo_classes(tmp_path):
    from vision.activity import ActivityVisionAdapter

    ckpt, _, _ = _train_tiny(tmp_path)
    adapter = ActivityVisionAdapter(ckpt)
    assert adapter.name == "activity"
    for cat in ("potential_fight", "potential_fall", "potential_weapon_object", "ordinary"):
        dets = adapter.detect(render_demo_frame(cat, seed=21), frame_index=0)
        labels = {d.label for d in dets}
        assert cat in labels
        top = next(d for d in dets if d.label == cat)
        assert 0.0 < top.confidence <= 1.0
        assert "scores" in top.extras


def test_create_adapter_uses_checkpoint(tmp_path, monkeypatch):
    ckpt, _, _ = _train_tiny(tmp_path)
    monkeypatch.setenv("VISION_BACKEND", "auto")
    monkeypatch.setenv("ACTIVITY_CHECKPOINT", str(ckpt))
    adapter = create_adapter()
    assert adapter.name == "activity"


def test_create_adapter_mock_wins_over_checkpoint(tmp_path, monkeypatch):
    ckpt, _, _ = _train_tiny(tmp_path)
    monkeypatch.setenv("VISION_BACKEND", "mock")
    monkeypatch.setenv("ACTIVITY_CHECKPOINT", str(ckpt))
    adapter = create_adapter("mock")
    assert adapter.name == "mock"


def test_create_adapter_activity_missing_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_BACKEND", "activity")
    monkeypatch.setenv("ACTIVITY_CHECKPOINT", str(tmp_path / "missing.joblib"))
    adapter = create_adapter("activity")
    assert adapter.name == "mock"


def test_create_adapter_auto_without_checkpoint_is_mock_or_yolo(tmp_path, monkeypatch):
    monkeypatch.setenv("VISION_BACKEND", "auto")
    monkeypatch.setenv("ACTIVITY_CHECKPOINT", str(tmp_path / "missing.joblib"))
    adapter = create_adapter("auto")
    assert adapter.name in {"mock", "yolo"}
