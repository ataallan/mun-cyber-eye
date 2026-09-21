"""Integration: MOCK vision → risk → alerts without YOLO."""

import numpy as np

from alerts.notify import NotificationService, NotifyConfig
from alerts.store import AlertStore
from ingest.sampler import SampledFrame
from pipeline import CyberEyePipeline, demo_synthetic_run
from vision.detector import Detection, VisionAdapter


def test_synthetic_demo_creates_alerts(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    result = demo_synthetic_run(store, frames=16)
    assert result.backend == "mock"
    assert result.source_label == "Authorized Camera — Synthetic Demo"
    assert result.frames_processed == 16
    assert len(result.alerts_created) >= 1
    open_alerts = store.list_alerts(status="open")
    assert len(open_alerts) == len(result.alerts_created)
    categories = {a.category for a in result.alerts_created}
    assert categories & {
        "potential_fight",
        "potential_fall",
        "potential_weapon_object",
    }
    first = result.alerts_created[0]
    assert first.severity in {"warning", "critical", "info"}
    assert first.correlation_id
    assert first.delivery_status == "pending"
    ids = {a.correlation_id for a in result.alerts_created}
    assert len(ids) == 1


def test_synthetic_demo_offline_notify_does_not_invent_success(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    store.add_recipient("op@example.com", created_by="test")
    notifier = NotificationService(
        store,
        NotifyConfig(resend_api_key="", email_recipients_env="op@example.com"),
    )
    result = demo_synthetic_run(store, frames=16, notifier=notifier)
    assert len(result.alerts_created) >= 1
    for alert in result.alerts_created:
        loaded = store.get(alert.id)
        assert loaded.delivery_status == "queued"
        assert loaded.delivery_status != "sent"


class _OrdinaryAdapter(VisionAdapter):
    name = "ordinary-test"

    def detect(self, image_bgr, frame_index: int = 0):
        return [Detection("ordinary", 0.94)]


def test_ordinary_frames_are_counted_without_alerts(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    pipe = CyberEyePipeline(
        store=store,
        adapter=_OrdinaryAdapter(),
        snapshot_dir=tmp_path / "snaps",
    )
    frames = [
        SampledFrame(
            index=i,
            timestamp_sec=i / 2.0,
            image_bgr=np.zeros((8, 8, 3), dtype=np.uint8),
            source_label="authorized upload — clip.avi",
        )
        for i in range(5)
    ]
    result = pipe.run_frames(frames, source_label="authorized upload — clip.avi")
    assert result.frames_processed == 5
    assert result.alerts_created == []
    assert result.category_counts == {"ordinary": 5}
    assert store.list_alerts() == []
