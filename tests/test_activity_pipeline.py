"""Pipeline uses the Phase 3 activity model when a checkpoint exists."""

from alerts.store import AlertStore
from pipeline import demo_activity_run
from vision.dataset import generate_demo_dataset
from vision.train_activity import train_activity_model


def test_activity_pipeline_creates_alerts(tmp_path):
    data_root = tmp_path / "activity"
    ckpt = tmp_path / "activity_demo.joblib"
    generate_demo_dataset(data_root, n_train=8, n_val=0, n_test=4, seed=3, overwrite=True)
    train_activity_model(
        data_root=data_root,
        output=ckpt,
        generate_demo=False,
        n_train=8,
        n_val=0,
        n_test=0,
    )
    store = AlertStore(tmp_path / "alerts.db")
    result = demo_activity_run(store, checkpoint=ckpt, per_class=2)
    assert result.backend == "activity"
    assert result.source_label == "Authorized Camera — Activity Demo"
    assert result.frames_processed == 8  # 4 classes × 2
    assert len(result.alerts_created) >= 1
    categories = {a.category for a in result.alerts_created}
    assert categories & {
        "potential_fight",
        "potential_fall",
        "potential_weapon_object",
    }
    assert "ordinary" not in categories
    meta = result.alerts_created[0].to_dict()["metadata"]
    assert meta.get("vision_backend") == "activity"
    assert "activity_checkpoint" in meta


def test_activity_pipeline_falls_back_to_mock(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    result = demo_activity_run(
        store, checkpoint=tmp_path / "no-such.joblib", per_class=2
    )
    assert result.backend == "mock"
    assert result.frames_processed == 8
