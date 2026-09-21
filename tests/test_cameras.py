"""Camera registry CRUD, URI masking, and demo seed."""

import os

import pytest

from ingest.cameras import (
    CameraStore,
    camera_from_form,
    is_mock_uri,
    mask_uri,
    resolve_uri,
)


@pytest.fixture
def store(tmp_path):
    return CameraStore(tmp_path / "cameras.db")


def test_create_get_list_and_disable(store):
    cam = store.create(
        name="Lobby west",
        location_label="Main entrance",
        source_type="file",
        uri="MOCK",
        sample_fps=2.0,
        notes="Authorized demo",
    )
    loaded = store.get(cam.id)
    assert loaded is not None
    assert loaded.name == "Lobby west"
    assert loaded.location_label == "Main entrance"
    assert loaded.place_type == ""
    assert loaded.source_type == "file"
    assert loaded.enabled is True
    assert loaded.effective_fps() == pytest.approx(2.0)

    store.set_enabled(cam.id, False)
    assert store.get(cam.id).enabled is False
    assert store.list_cameras(enabled_only=True) == []
    assert len(store.list_cameras()) == 1


def test_update_keeps_uri_when_omitted(store):
    cam = store.create(
        name="NVR",
        source_type="rtsp",
        uri="rtsp://op:s3cret@nvr.example/stream",
        location_label="Yard",
    )
    updated = store.update(cam.id, name="Yard NVR")
    assert updated.name == "Yard NVR"
    assert updated.uri == "rtsp://op:s3cret@nvr.example/stream"


def test_sample_interval_overrides_fps(store):
    cam = store.create(
        name="Slow cam",
        source_type="file",
        uri="MOCK",
        sample_fps=10.0,
        sample_interval=0.5,
    )
    assert cam.effective_fps() == pytest.approx(2.0)


def test_record_success_clears_error(store):
    cam = store.create(name="Lab", source_type="file", uri="MOCK")
    store.record_error(cam.id, "offline")
    assert store.get(cam.id).last_error == "offline"
    store.record_success(cam.id)
    refreshed = store.get(cam.id)
    assert refreshed.last_error is None
    assert refreshed.last_seen_at


def test_invalid_source_type(store):
    with pytest.raises(ValueError):
        store.create(name="Bad", source_type="drone", uri="x")


def test_place_type_round_trip_and_reject_unknown(store):
    cam = store.create(
        name="North hall",
        location_label="North corridor",
        source_type="file",
        uri="MOCK",
        place_type="corridor_hallway",
    )
    assert cam.place_type == "corridor_hallway"
    assert store.get(cam.id).place_type == "corridor_hallway"
    updated = store.update(cam.id, place_type="house_interior")
    assert updated.place_type == "house_interior"
    with pytest.raises(ValueError):
        store.create(name="Arena", source_type="file", uri="MOCK", place_type="madison_square_garden")


def test_mask_and_resolve_uri(monkeypatch):
    assert mask_uri("rtsp://admin:hunter2@cam.example:554/h264") == (
        "rtsp://admin:****@cam.example:554/h264"
    )
    assert mask_uri("env:RTSP_DEMO_URI") == "env:RTSP_DEMO_URI"
    assert mask_uri("MOCK") == "MOCK"
    assert mask_uri("sample_data/demo.mp4") == "sample_data/demo.mp4"
    monkeypatch.setenv("RTSP_DEMO_URI", "rtsp://u:p@host/s")
    assert resolve_uri("env:RTSP_DEMO_URI") == "rtsp://u:p@host/s"
    monkeypatch.delenv("RTSP_DEMO_URI", raising=False)
    assert resolve_uri("env:RTSP_DEMO_URI") == ""
    assert is_mock_uri("MOCK")
    public = store_public_uri()
    assert "hunter2" not in public
    assert "****" in public


def store_public_uri():
    from ingest.cameras import Camera

    cam = Camera(
        id="x",
        name="n",
        location_label="l",
        source_type="rtsp",
        uri="rtsp://admin:hunter2@cam.example/s",
        enabled=True,
        sample_fps=2.0,
        sample_interval=None,
        notes="",
        created_at="t",
        updated_at="t",
    )
    return cam.to_public_dict()["uri"]


def test_seed_demo_cameras_once(store, tmp_path):
    seeded = store.seed_demo_cameras(tmp_path)
    assert {c.id for c in seeded} == {"demo-file-01", "demo-rtsp-01"}
    file_cam = store.get("demo-file-01")
    rtsp_cam = store.get("demo-rtsp-01")
    assert file_cam.enabled is True
    assert file_cam.source_type == "file"
    assert file_cam.place_type == "gymnasium"
    assert rtsp_cam.enabled is False
    assert rtsp_cam.place_type == "street"
    assert rtsp_cam.uri == "env:RTSP_DEMO_URI"
    again = store.seed_demo_cameras(tmp_path)
    assert len(again) == 2


def test_camera_from_form_blank_uri_keeps_existing(store):
    cam = store.create(
        name="Keep",
        source_type="rtsp",
        uri="rtsp://u:p@h/s",
        enabled=True,
    )

    class Form(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    payload = camera_from_form(
        Form(
            {
                "name": "Keep",
                "location_label": "Dock",
                "source_type": "rtsp",
                "uri": "",
                "sample_fps": "2",
                "enabled": "1",
                "notes": "",
                "place_type": "street",
            }
        ),
        existing=cam,
    )
    assert "uri" not in payload
    assert payload["place_type"] == "street"
    updated = store.update(cam.id, **payload)
    assert updated.uri == "rtsp://u:p@h/s"
    assert updated.location_label == "Dock"


def test_health_omits_plaintext_password(store):
    store.create(
        name="Secret cam",
        source_type="rtsp",
        uri="rtsp://op:super-secret@nvr.example/stream",
    )
    health = store.health()
    blob = str(health)
    assert "super-secret" not in blob
    assert os.getenv("super-secret") is None
