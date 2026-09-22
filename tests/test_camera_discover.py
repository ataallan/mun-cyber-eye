"""Local webcam discovery, registry dedupe, and the Detect cameras route."""

import time

import pytest

from app.factory import create_app
from ingest.cameras import CameraStore
from ingest.discover import (
    DiscoveredCamera,
    _call_with_timeout,
    add_discovered_webcams,
    discover_local_cameras,
    discover_webcams,
    webcam_index_from_uri,
)


class _Cap:
    def __init__(self, ok: bool = True, frame: bool = True):
        self.ok = ok
        self.frame = frame
        self.released = False

    def isOpened(self):
        return self.ok

    def read(self):
        if not self.frame:
            return False, None
        return True, b"frame"

    def release(self):
        self.released = True


def _opener(good: set[int], *, frame: bool = True, seen: list | None = None):
    def opener(index: int):
        if seen is not None:
            seen.append(index)
        if index not in good:
            return None
        return _Cap(ok=True, frame=frame)

    return opener


def test_webcam_index_from_uri_matches_registry_convention():
    assert webcam_index_from_uri("0") == 0
    assert webcam_index_from_uri(" 2 ") == 2
    assert webcam_index_from_uri("/dev/video1") == 1
    assert webcam_index_from_uri("") is None
    assert webcam_index_from_uri("rtsp://cam/stream") is None
    assert webcam_index_from_uri("env:MISSING_WEBCAM") is None


def test_discover_webcams_names_open_devices_and_skips_failures():
    seen: list[int] = []
    found = discover_webcams(
        opener=_opener({0, 1}, seen=seen),
        max_index=8,
        miss_streak=3,
    )
    assert [(item.name, item.source_type, item.uri) for item in found] == [
        ("Webcam 0", "webcam", "0"),
        ("Webcam 1", "webcam", "1"),
    ]
    assert seen == [0, 1, 2, 3, 4]
    assert all(item.already_registered is False for item in found)


def test_discover_stops_after_miss_streak_and_ignores_opener_errors():
    seen: list[int] = []

    def opener(index: int):
        seen.append(index)
        if index == 0:
            raise RuntimeError("device busy")
        return None

    found = discover_webcams(opener=opener, max_index=8, miss_streak=3)
    assert found == []
    assert seen == [0, 1, 2]


def test_discover_requires_a_frame_unless_disabled():
    blind = discover_webcams(
        opener=_opener({0}, frame=False),
        max_index=1,
        miss_streak=1,
        grab_frame=True,
    )
    assert blind == []
    opened = discover_webcams(
        opener=_opener({0}, frame=False),
        max_index=1,
        miss_streak=1,
        grab_frame=False,
    )
    assert [item.uri for item in opened] == ["0"]


def test_discover_marks_existing_webcam_index_and_ignores_other_sources(tmp_path):
    store = CameraStore(tmp_path / "cameras.db")
    store.create(name="Lobby", source_type="webcam", uri="0")
    store.create(name="Door", source_type="webcam", uri=" /dev/video1 ")
    store.create(name="Clip", source_type="file", uri="2")
    store.create(name="Gate", source_type="rtsp", uri="rtsp://nvr/stream")
    found = discover_webcams(
        store.list_cameras(),
        opener=_opener({0, 1, 2}),
        max_index=4,
        miss_streak=2,
    )
    by_uri = {item.uri: item for item in found}
    assert by_uri["0"].already_registered is True
    assert by_uri["1"].already_registered is True
    assert by_uri["2"].already_registered is False
    assert by_uri["2"].name == "Webcam 2"


def test_env_webcam_uri_counts_as_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("LAB_WEBCAM", "1")
    store = CameraStore(tmp_path / "cameras.db")
    store.create(name="Lab", source_type="webcam", uri="env:LAB_WEBCAM")
    found = discover_webcams(
        store.list_cameras(),
        opener=_opener({0, 1}),
        max_index=3,
        miss_streak=2,
    )
    by_uri = {item.uri: item.already_registered for item in found}
    assert by_uri == {"0": False, "1": True}


def test_add_discovered_dedupes_and_uses_seed_defaults(tmp_path):
    store = CameraStore(tmp_path / "cameras.db")
    store.create(name="Lobby", source_type="webcam", uri="0", enabled=False)
    found = [
        DiscoveredCamera("Webcam 0", "webcam", "0", already_registered=True),
        DiscoveredCamera("Webcam 2", "webcam", "2"),
        DiscoveredCamera("Webcam 2", "webcam", "2"),
        DiscoveredCamera("Network", "rtsp", "rtsp://nvr/stream"),
    ]
    created = add_discovered_webcams(store, found)
    assert [c.name for c in created] == ["Webcam 2"]
    added = store.get(created[0].id)
    assert added is not None
    assert added.source_type == "webcam"
    assert added.uri == "2"
    assert added.enabled is True
    assert added.sample_fps == 2.0
    again = add_discovered_webcams(store, found)
    assert again == []
    selected = add_discovered_webcams(
        store,
        [DiscoveredCamera("Webcam 3", "webcam", "3")],
        only_uris=["0", "nope"],
    )
    assert selected == []
    manual = store.create(
        name="Gate",
        source_type="rtsp",
        uri="rtsp://admin:secret@nvr/stream",
        enabled=True,
    )
    assert manual.source_type == "rtsp"


def test_hung_index_is_skipped_quickly(monkeypatch):
    class _Hung:
        def set(self, *_args, **_kwargs):
            return None

        def open(self, *_args, **_kwargs):
            time.sleep(0.8)
            return False

        def isOpened(self):
            return False

        def release(self):
            return None

    monkeypatch.setattr("cv2.VideoCapture", lambda *_a, **_k: _Hung())
    started = time.perf_counter()
    found = discover_webcams(
        max_index=2,
        miss_streak=1,
        open_timeout_sec=0.15,
    )
    elapsed = time.perf_counter() - started
    assert found == []
    assert elapsed < 0.6


def test_default_opener_accepts_a_fake_capture(monkeypatch):
    class _Good:
        def __init__(self):
            self.released = False

        def set(self, *_args, **_kwargs):
            return None

        def open(self, index, backend=None):
            return int(index) == 0

        def isOpened(self):
            return True

        def read(self):
            return True, b"frame"

        def release(self):
            self.released = True

    holder: dict = {}

    def factory(*_args, **_kwargs):
        cap = _Good()
        holder["cap"] = cap
        return cap

    monkeypatch.setattr("cv2.VideoCapture", factory)
    found = discover_local_cameras(max_index=1, miss_streak=1, open_timeout_sec=1.0)
    assert [(item.name, item.uri, item.source_type) for item in found] == [
        ("Webcam 0", "0", "webcam")
    ]
    assert holder["cap"].released is True


def test_call_with_timeout_returns_fast_results_and_abandons_slow_ones():
    assert _call_with_timeout(lambda: "ok", 0.5) == "ok"

    def fail():
        raise RuntimeError("unavailable")

    assert _call_with_timeout(fail, 0.5) is None
    started = time.perf_counter()
    assert _call_with_timeout(lambda: time.sleep(0.6), 0.1) is None
    assert time.perf_counter() - started < 0.4


@pytest.fixture
def app(tmp_path):
    return create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
            "ALLOW_WEBCAM": False,
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client):
    return client.post(
        "/login",
        data={"username": "siteadmin", "password": "test-pass-12"},
        follow_redirects=True,
    )


def _enable_detect(app, good: set[int]):
    app.config["ALLOW_WEBCAM"] = True
    app.config["WEBCAM_DISCOVERY_OPENER"] = _opener(good)


def test_detect_requires_same_login_as_add_camera(client):
    detect = client.get("/cameras/detect")
    add = client.get("/cameras/new")
    assert detect.status_code == 302
    assert add.status_code == 302
    assert "/login" in detect.headers["Location"]
    assert "/login" in add.headers["Location"]


def test_detect_hidden_when_webcam_capture_is_off(client):
    _login(client)
    page = client.get("/cameras")
    body = page.get_data(as_text=True)
    assert "Add camera" in body
    assert "Detect cameras" not in body
    assert "Webcam off" in body
    blocked = client.get("/cameras/detect", follow_redirects=True)
    assert "Webcam capture is off." in blocked.get_data(as_text=True)
    assert "Detected cameras" not in blocked.get_data(as_text=True)


def test_detect_route_lists_and_registers_new_webcams_only(app, client):
    _login(client)
    _enable_detect(app, {0, 2})
    app.extensions["camera_store"].create(
        name="Lobby",
        source_type="webcam",
        uri="0",
        enabled=True,
    )
    listing = client.get("/cameras")
    listing_body = listing.get_data(as_text=True)
    assert 'href="/cameras/detect"' in listing_body
    assert "Detect cameras" in listing_body
    assert "Add camera" in listing_body

    form = client.get("/cameras/new")
    form_body = form.get_data(as_text=True)
    assert 'value="file"' in form_body
    assert 'value="rtsp"' in form_body
    assert 'value="webcam"' in form_body

    detected = client.get("/cameras/detect")
    body = detected.get_data(as_text=True)
    assert detected.status_code == 200
    assert "Detected cameras" in body
    assert "Webcam 0" in body
    assert "Webcam 2" in body
    assert "registered" in body
    assert 'value="2"' in body
    assert 'name="indices"' in body

    empty = client.post("/cameras/detect", data={"action": "selected"}, follow_redirects=True)
    assert "Select a camera to add." in empty.get_data(as_text=True)
    names = {c.uri for c in app.extensions["camera_store"].list_cameras() if c.source_type == "webcam"}
    assert names == {"0"}

    added = client.post(
        "/cameras/detect",
        data={"action": "selected", "indices": "2"},
        follow_redirects=True,
    )
    added_body = added.get_data(as_text=True)
    assert "Webcam 2" in added_body
    assert "Camera “Webcam 2” registered." in added_body
    stored = [
        c
        for c in app.extensions["camera_store"].list_cameras()
        if c.source_type == "webcam"
    ]
    by_uri = {c.uri: c for c in stored}
    assert set(by_uri) == {"0", "2"}
    assert by_uri["0"].name == "Lobby"
    assert by_uri["2"].enabled is True
    assert by_uri["2"].sample_fps == 2.0

    duplicate = client.post(
        "/cameras/detect",
        data={"action": "all_new"},
        follow_redirects=True,
    )
    assert "No new cameras to add." in duplicate.get_data(as_text=True)
    webcam_rows = [
        c
        for c in app.extensions["camera_store"].list_cameras()
        if c.source_type == "webcam" and c.uri in {"0", "2"}
    ]
    assert len(webcam_rows) == 2

    manual = client.post(
        "/cameras/new",
        data={
            "name": "Gate RTSP",
            "location_label": "North gate",
            "source_type": "rtsp",
            "uri": "rtsp://nvr.example/stream1",
            "sample_fps": "2",
            "enabled": "1",
        },
        follow_redirects=True,
    )
    assert manual.status_code == 200
    assert "Gate RTSP" in manual.get_data(as_text=True)


def test_add_all_new_registers_every_unlisted_webcam(app, client):
    _login(client)
    _enable_detect(app, {0, 1})
    resp = client.post(
        "/cameras/detect",
        data={"action": "all_new", "indices": ""},
        follow_redirects=True,
    )
    assert "Registered 2 cameras." in resp.get_data(as_text=True)
    stored = {
        c.uri: c
        for c in app.extensions["camera_store"].list_cameras()
        if c.source_type == "webcam"
    }
    assert set(stored) == {"0", "1"}
    assert stored["0"].name == "Webcam 0"
    assert stored["1"].enabled is True
