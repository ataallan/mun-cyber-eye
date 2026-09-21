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


def test_owner_fields_round_trip_and_many_accounts(store):
    cam = store.create(
        name="Owned cam",
        source_type="file",
        uri="MOCK",
        owner_user_id="user-1",
        owner_username="alice",
        notify_email="OnCall@Site.example, alice@site.example, not-an-email",
    )
    loaded = store.get(cam.id)
    assert loaded.owner_user_id == "user-1"
    assert loaded.owner_username == "alice"
    assert loaded.notify_email == "oncall@site.example, alice@site.example"
    assert store.list_accounts_for_camera(cam.id)[0]["user_id"] == "user-1"
    store.set_accounts_for_camera(
        cam.id, [("user-1", "alice"), ("user-2", "bob")]
    )
    ids = {row["user_id"] for row in store.list_accounts_for_camera(cam.id)}
    assert ids == {"user-1", "user-2"}
    store.set_accounts_for_camera(cam.id, [])
    store.update(cam.id, notify_email="")
    assert store.list_accounts_for_camera(cam.id) == []
    assert store.get(cam.id).notify_email == ""


def test_one_account_many_cameras(store):
    a = store.create(name="Cam A", source_type="file", uri="MOCK")
    b = store.create(name="Cam B", source_type="file", uri="MOCK")
    store.set_cameras_for_user("user-9", "pat", [a.id, b.id])
    assert set(store.list_camera_ids_for_user("user-9")) == {a.id, b.id}
    store.set_cameras_for_user("user-9", "pat", [b.id])
    assert store.list_camera_ids_for_user("user-9") == [b.id]
    assert store.list_accounts_for_camera(a.id) == []
    assert store.list_accounts_for_camera(b.id)[0]["username"] == "pat"


def test_camera_from_form_resolves_owner(store):
    class Form(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _User:
        def __init__(self):
            self.id = "uid-9"
            self.username = "pat"
            self.email = "pat@example.com"

    class _Users:
        def get_by_id(self, user_id):
            return _User() if user_id == "uid-9" else None

    payload = camera_from_form(
        Form(
            {
                "name": "Gate",
                "location_label": "North",
                "source_type": "file",
                "uri": "MOCK",
                "sample_fps": "2",
                "enabled": "1",
                "notes": "",
                "place_type": "street",
                "owner_user_id": "uid-9",
                "notify_email": "extra@example.com",
            }
        ),
        user_store=_Users(),
    )
    assert payload["owner_user_id"] == "uid-9"
    assert payload["owner_username"] == "pat"
    assert payload["notify_email"] == "extra@example.com"
    created = store.create(**payload)
    assert created.owner_username == "pat"


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
    custom = store.create(
        name="Arena cam",
        source_type="file",
        uri="MOCK",
        place_type="madison_square_garden",
    )
    assert custom.place_type == "madison_square_garden"


def test_custom_place_rooftop_cafe_round_trip(store):
    cam = store.create(
        name="Roof cam",
        location_label="Level 4",
        source_type="file",
        uri="MOCK",
        notes="Authorized rooftop",
        place_type="rooftop café",
    )
    assert cam.place_type == "rooftop_cafe"
    assert store.get(cam.id).place_type == "rooftop_cafe"
    listed = store.list_cameras()
    assert any(c.place_type == "rooftop_cafe" for c in listed)
    ids = {p.id for p in store.list_place_choices()}
    assert "rooftop_cafe" in ids
    assert "roam" in ids
    updated = store.update(cam.id, place_type="warehouse bay 3")
    assert updated.place_type == "warehouse_bay_3"
    assert any(p.id == "warehouse_bay_3" for p in store.list_custom_places())


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


EXPECTED_DEMO_PLACES = {
    "demo-file-01": "gymnasium",
    "demo-rtsp-01": "street",
    "demo-corridor-01": "corridor_hallway",
    "demo-house-01": "house_interior",
    "demo-compound-01": "compound_courtyard",
    "demo-roam-01": "roam",
}


def test_seed_demo_cameras_once(store, tmp_path):
    seeded = store.seed_demo_cameras(tmp_path)
    by_id = {c.id: c for c in seeded}
    assert set(by_id) == set(EXPECTED_DEMO_PLACES)
    file_cam = by_id["demo-file-01"]
    rtsp_cam = by_id["demo-rtsp-01"]
    assert file_cam.enabled is True
    assert file_cam.source_type == "file"
    assert file_cam.place_type == "gymnasium"
    assert rtsp_cam.enabled is False
    assert rtsp_cam.place_type == "street"
    assert rtsp_cam.uri == "env:RTSP_DEMO_URI"
    assert by_id["demo-corridor-01"].name == "Corridor North"
    assert by_id["demo-corridor-01"].place_type == "corridor_hallway"
    assert by_id["demo-house-01"].name == "House interior demo"
    assert by_id["demo-house-01"].place_type == "house_interior"
    assert by_id["demo-compound-01"].name == "Compound courtyard"
    assert by_id["demo-compound-01"].place_type == "compound_courtyard"
    assert by_id["demo-roam-01"].name == "Roam / patrol cam"
    assert by_id["demo-roam-01"].place_type == "roam"
    assert all(c.place_type == EXPECTED_DEMO_PLACES[c.id] for c in seeded)
    assert all(c.owner_user_id == "" and c.owner_username == "" for c in seeded)
    for cam in seeded:
        if cam.id != "demo-rtsp-01":
            assert cam.enabled is True
            assert cam.source_type == "file"
        assert "uthorized" in cam.notes
    again = store.seed_demo_cameras(tmp_path)
    assert len(again) == 6


def test_list_and_seed_backfill_empty_demo_place_types(store, tmp_path):
    store.create(
        camera_id="demo-file-01",
        name="Demo Lab File",
        location_label="Demo Lab",
        source_type="file",
        uri="MOCK",
        notes="Authorized demo",
        place_type="",
    )
    store.create(
        camera_id="demo-rtsp-01",
        name="Authorized RTSP stub",
        location_label="Perimeter (stub)",
        source_type="rtsp",
        uri="env:RTSP_DEMO_URI",
        enabled=False,
        notes="Authorized demo",
        place_type="",
    )
    listed = store.list_cameras()
    assert store.get("demo-file-01").place_type == "gymnasium"
    assert store.get("demo-rtsp-01").place_type == "street"
    assert {c.id for c in listed} == {"demo-file-01", "demo-rtsp-01"}

    seeded = store.seed_demo_cameras(tmp_path)
    by_id = {c.id: c for c in seeded}
    assert by_id["demo-file-01"].place_type == "gymnasium"
    assert by_id["demo-rtsp-01"].place_type == "street"
    assert by_id["demo-rtsp-01"].enabled is False
    assert by_id["demo-corridor-01"].place_type == "corridor_hallway"
    assert by_id["demo-house-01"].place_type == "house_interior"
    assert by_id["demo-compound-01"].place_type == "compound_courtyard"
    assert by_id["demo-roam-01"].place_type == "roam"


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
    custom_payload = camera_from_form(
        Form(
            {
                "name": "Keep",
                "location_label": "Dock",
                "source_type": "rtsp",
                "uri": "",
                "sample_fps": "2",
                "enabled": "1",
                "notes": "",
                "place_type": "__custom__",
                "place_type_custom": "rooftop café",
            }
        ),
        existing=cam,
    )
    assert custom_payload["place_type"] == "rooftop café"
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
