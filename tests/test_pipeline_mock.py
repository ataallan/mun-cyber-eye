"""Integration: MOCK vision → risk → alerts without YOLO."""

from alerts.store import AlertStore
from pipeline import demo_synthetic_run


def test_synthetic_demo_creates_alerts(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    result = demo_synthetic_run(store, frames=16)
    assert result.backend == "mock"
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
