"""Linked camera accounts + security_email on Resend delivery."""

import json

from alerts.notify import EmailAdapter, NotificationService, NotifyConfig, WebhookAdapter
from alerts.store import AlertStore
from app.auth import UserStore
from app.factory import create_app
from ingest.cameras import CameraStore


class _FakeResp:
    def __init__(self, status, body):
        self.status = status
        self._body = body.encode("utf-8")

    def read(self):
        return self._body

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _alert(store, camera_id="cam-owned", **kwargs):
    defaults = dict(
        source_label="Linked Camera",
        category="potential_fight",
        risk_level="high",
        confidence=0.8,
        rationale="Possible fight — verify.",
        frame_index=1,
        timestamp_sec=1.0,
        location_label="North hall",
        camera_id=camera_id,
    )
    defaults.update(kwargs)
    return store.create_alert(**defaults)


def _service(store, cameras, users, seen, env="", security_fallback=""):
    def opener(req, timeout=15):
        seen["url"] = req.full_url
        seen["body"] = json.loads(req.data.decode("utf-8"))
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        return _FakeResp(200, json.dumps({"id": "re_owner"}))

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env=env,
        security_alert_email=security_fallback,
        max_attempts=1,
        sleep_fn=lambda _s: None,
    )
    return NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg),
        camera_store=cameras,
        user_store=users,
    )


def test_deliver_includes_all_linked_account_emails(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    owner = users.create_user(
        username="camowner",
        email="owner@example.com",
        password="test-pass-12",
    )
    users.set_security_email(owner.id, "security@example.com")
    extra = users.create_user(
        username="reviewer",
        email="reviewer@example.com",
        password="test-pass-12",
    )
    cameras.create(
        camera_id="cam-owned",
        name="Lobby west",
        location_label="Main lobby",
        source_type="file",
        uri="MOCK",
        place_type="corridor_hallway",
        notify_email="extra-cam@example.com",
    )
    cameras.set_accounts_for_camera(
        "cam-owned", [(owner.id, owner.username), (extra.id, extra.username)]
    )
    seen = {}
    svc = _service(store, cameras, users, seen, env="global@example.com")
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "sent"
    dest = seen["body"]["to"]
    assert dest[0] in {"security@example.com", "reviewer@example.com"}
    assert "security@example.com" in dest
    assert "owner@example.com" not in dest
    assert "reviewer@example.com" in dest
    assert "extra-cam@example.com" in dest
    assert "global@example.com" in dest
    assert "Lobby west" in seen["body"]["subject"]
    meta = updated.to_dict()["metadata"]
    assert meta["notified_emails"] == dest


def test_one_account_covers_many_cameras(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    owner = users.create_user(
        username="siteops",
        email="ops@example.com",
        password="test-pass-12",
    )
    users.set_security_email(owner.id, "security@example.com")
    cameras.create(camera_id="cam-a", name="A", source_type="file", uri="MOCK")
    cameras.create(camera_id="cam-b", name="B", source_type="file", uri="MOCK")
    cameras.set_cameras_for_user(owner.id, owner.username, ["cam-a", "cam-b"])
    seen_a, seen_b = {}, {}
    _service(store, cameras, users, seen_a).deliver(_alert(store, "cam-a"))
    _service(store, cameras, users, seen_b).deliver(_alert(store, "cam-b"))
    assert seen_a["body"]["to"][0] == "security@example.com"
    assert seen_b["body"]["to"][0] == "security@example.com"


def test_deliver_no_account_uses_env_and_directory_only(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    cameras.create(camera_id="cam-owned", name="Unowned", source_type="file", uri="MOCK")
    store.add_recipient("directory@example.com", created_by="admin")
    seen = {}
    svc = _service(store, cameras, users, seen, env="env@example.com")
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "sent"
    assert seen["body"]["to"] == ["env@example.com", "directory@example.com"]


def test_security_fallback_only_when_no_account_and_no_session(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    cameras.create(camera_id="cam-owned", name="Bare", source_type="file", uri="MOCK")
    seen = {}
    svc = _service(
        store,
        cameras,
        users,
        seen,
        env="",
        security_fallback="security@example.com",
    )
    svc.deliver(_alert(store))
    assert seen["body"]["to"] == ["security@example.com"]

    owner = users.create_user(
        username="linked",
        email="linked@example.com",
        password="test-pass-12",
    )
    cameras.set_accounts_for_camera("cam-owned", [(owner.id, owner.username)])
    seen2 = {}
    _service(
        store,
        cameras,
        users,
        seen2,
        env="",
        security_fallback="security@example.com",
    ).deliver(_alert(store))
    assert seen2["body"]["to"] == ["linked@example.com"]
    assert "security@example.com" not in seen2["body"]["to"]


def test_owner_without_email_skipped_with_honest_note(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    owner = users.create_user(
        username="noemail",
        email="placeholder@example.com",
        password="test-pass-12",
    )
    with users._conn() as conn:
        conn.execute("UPDATE users SET email = '', security_email = '' WHERE id = ?", (owner.id,))
    cameras.create(
        camera_id="cam-owned",
        name="Bare owner",
        source_type="file",
        uri="MOCK",
        owner_user_id=owner.id,
        owner_username=owner.username,
    )
    seen = {}
    svc = _service(store, cameras, users, seen, env="env@example.com")
    updated = svc.deliver(_alert(store))
    assert seen["body"]["to"] == ["env@example.com"]
    notes = " ".join(updated.to_dict()["metadata"].get("notify_notes") or [])
    assert "email" in notes.lower()


def test_signed_in_extra_recipient_from_route(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "actor@example.com",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
            "ALERT_EMAIL_RECIPIENTS": "",
            "SEED_DEMO_CAMERAS": False,
        }
    )
    seen = {}

    def opener(req, timeout=15):
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(200, json.dumps({"id": "re_actor"}))

    store = application.extensions["alert_store"]
    cameras = application.extensions["camera_store"]
    users = application.extensions["user_store"]
    users.set_security_email(
        users.get_by_username("siteadmin").id, "actor-sec@example.com"
    )
    owner = users.create_user(
        username="siteowner",
        email="owner@example.com",
        password="test-pass-12",
    )
    cameras.create(
        camera_id="cam-route",
        name="Route cam",
        location_label="Dock",
        source_type="file",
        uri="MOCK",
    )
    cameras.set_accounts_for_camera("cam-route", [(owner.id, owner.username)])
    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="",
        max_attempts=1,
        sleep_fn=lambda _s: None,
        enabled=True,
    )
    application.extensions["notify_config"] = cfg
    application.extensions["notifier"] = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg),
        camera_store=cameras,
        user_store=users,
    )
    alert = store.create_alert(
        source_label="Route cam",
        category="potential_fall",
        risk_level="elevated",
        confidence=0.6,
        rationale="Possible fall — verify.",
        frame_index=1,
        timestamp_sec=1.0,
        camera_id="cam-route",
        location_label="Dock",
    )
    client = application.test_client()
    client.post("/login", data={"username": "siteadmin", "password": "test-pass-12"})
    resp = client.post(f"/alerts/{alert.id}/notify", follow_redirects=True)
    assert resp.status_code == 200
    dest = seen["body"]["to"]
    assert dest[0] == "owner@example.com"
    assert "actor-sec@example.com" in dest
    assert "actor@example.com" not in dest
    loaded = store.get(alert.id)
    assert loaded.delivery_status == "sent"


def test_account_page_attaches_many_cameras(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "ops@example.com",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
        }
    )
    client = application.test_client()
    client.post("/login", data={"username": "siteadmin", "password": "test-pass-12"})
    mine = client.get("/account")
    body = mine.get_data(as_text=True)
    assert mine.status_code == 200
    assert "My cameras" in body
    assert "Security email" in body
    assert "demo-file-01" in body
    saved = client.post(
        "/account",
        data={
            "security_email": "security@example.com",
            "camera_ids": ["demo-file-01", "demo-corridor-01"],
        },
        follow_redirects=True,
    )
    assert saved.status_code == 200
    user = application.extensions["user_store"].get_by_username("siteadmin")
    assert user.security_email == "security@example.com"
    linked = set(
        application.extensions["camera_store"].list_camera_ids_for_user(user.id)
    )
    assert linked == {"demo-file-01", "demo-corridor-01"}
    listing = client.get("/accounts")
    assert listing.status_code == 200
    assert "siteadmin" in listing.get_data(as_text=True)
    assert "security@example.com" in listing.get_data(as_text=True)


def test_factory_wires_camera_and_user_stores_into_notifier(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "ops@example.com",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
        }
    )
    notifier = application.extensions["notifier"]
    assert notifier.camera_store is application.extensions["camera_store"]
    assert notifier.user_store is application.extensions["user_store"]
    admin = application.extensions["user_store"].get_by_username("siteadmin")
    assert admin.email == "ops@example.com"


def test_run_pipeline_includes_signed_in_account(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "actor@example.com",
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "CAMERA_DB_PATH": str(tmp_path / "cameras.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": True,
            "ALERT_EMAIL_RECIPIENTS": "",
            "SEED_DEMO_CAMERAS": True,
        }
    )
    seen = {"tos": []}

    def opener(req, timeout=15):
        body = json.loads(req.data.decode("utf-8"))
        if "resend.com" in req.full_url:
            seen["tos"].append(list(body.get("to") or []))
            seen["subject"] = body.get("subject")
        return _FakeResp(200, json.dumps({"id": "re_run"}))

    store = application.extensions["alert_store"]
    users = application.extensions["user_store"]
    users.set_security_email(
        users.get_by_username("siteadmin").id, "security@example.com"
    )
    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="",
        max_attempts=1,
        sleep_fn=lambda _s: None,
        enabled=True,
    )
    application.extensions["notify_config"] = cfg
    application.extensions["notifier"] = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg),
        camera_store=application.extensions["camera_store"],
        user_store=users,
    )
    client = application.test_client()
    client.post("/login", data={"username": "siteadmin", "password": "test-pass-12"})
    resp = client.post("/run", data={"mode": "synthetic"}, follow_redirects=True)
    assert resp.status_code == 200
    assert seen["tos"]
    for dest in seen["tos"]:
        assert "security@example.com" in dest
        assert "actor@example.com" not in dest


def test_missing_resend_key_with_linked_account_stays_queued(tmp_path):
    store = AlertStore(tmp_path / "alerts.db")
    cameras = CameraStore(tmp_path / "cameras.db")
    users = UserStore(tmp_path / "auth.db")
    owner = users.create_user(
        username="camowner",
        email="owner@example.com",
        password="test-pass-12",
    )
    cameras.create(
        camera_id="cam-owned",
        name="Lobby",
        source_type="file",
        uri="MOCK",
        owner_user_id=owner.id,
        owner_username=owner.username,
    )
    cfg = NotifyConfig(resend_api_key="", email_recipients_env="")
    svc = NotificationService(
        store, cfg, camera_store=cameras, user_store=users
    )
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "queued"
    assert "owner@example.com" in updated.to_dict()["metadata"]["notified_emails"]
    email_rows = [r for r in store.delivery_trail(updated.id) if r["channel"] == "email"]
    assert email_rows[-1]["status"] == "queued"
    assert "RESEND_API_KEY" in email_rows[-1]["error"]
