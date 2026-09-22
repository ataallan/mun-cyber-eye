"""Flask console: structured detail, recipients, resend, health."""

import json

import pytest

from alerts.notify import EmailAdapter, NotificationService, NotifyConfig, WebhookAdapter
from app.factory import create_app


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


@pytest.fixture
def app(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_ROLE": "admin",
            "OPERATOR_USERNAME": "reviewer",
            "OPERATOR_PASSWORD": "review",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snaps"),
            "VISION_BACKEND": "mock",
            "ALERT_NOTIFY_ON_CREATE": False,
            "RESEND_API_KEY": "",
            "ALERT_EMAIL_RECIPIENTS": "env.op@example.com",
        }
    )
    return application


@pytest.fixture
def client(app):
    return app.test_client()


def _login(client, username="siteadmin", password="test-pass-12"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=True,
    )


def _make_alert(app, **kwargs):
    store = app.extensions["alert_store"]
    defaults = dict(
        source_label="Authorized Camera — Test",
        category="potential_weapon_object",
        risk_level="high",
        confidence=0.77,
        rationale="Possible object — human verification required.",
        frame_index=4,
        timestamp_sec=2.25,
        location_label="Lobby",
        camera_id="cam-1",
    )
    defaults.update(kwargs)
    return store.create_alert(**defaults)


def test_health_reports_phase_5_and_honest_notify(client):
    data = client.get("/health").get_json()
    assert data["phase"] == 5
    assert data["notify"]["resend_configured"] is False
    assert data["notify"]["recipient_count"] >= 1
    assert "cameras" in data


def test_recipients_requires_auth(client):
    resp = client.get("/recipients")
    assert resp.status_code == 302
    assert "/login" in resp.headers["Location"]


def test_operator_can_add_recipient(app, client):
    _login(client)
    resp = client.post(
        "/recipients",
        data={"email": "oncall@example.com", "display_name": "On call", "role": "operator"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert b"oncall@example.com" in resp.data
    emails = [r["email"] for r in app.extensions["alert_store"].list_recipients()]
    assert "oncall@example.com" in emails


def test_alert_detail_shows_structured_fields(app, client):
    alert = _make_alert(app)
    _login(client)
    resp = client.get(f"/alerts/{alert.id}")
    assert resp.status_code == 200
    body = resp.data.decode("utf-8")
    assert "AI detects and alerts. Humans verify and decide." in body
    assert "critical" in body
    assert "Lobby" in body
    assert "cam-1" in body
    assert "correlation" in body.lower()
    assert "Recommended human action" in body
    assert "potential weapon-like object" in body
    assert "Resend to authorized recipients" in body
    payload = client.get(f"/alerts/{alert.id}.json").get_json()
    assert payload["camera_id"] == "cam-1"
    assert payload["severity"] == "critical"


def test_resend_without_key_stays_queued(app, client):
    alert = _make_alert(app)
    _login(client)
    resp = client.post(f"/alerts/{alert.id}/notify", follow_redirects=True)
    assert resp.status_code == 200
    loaded = app.extensions["alert_store"].get(alert.id)
    assert loaded.delivery_status == "queued"
    assert b"queued" in resp.data


def test_resend_success_updates_status(app, client):
    def opener(req, timeout=15):
        return _FakeResp(200, json.dumps({"id": "re_ui"}))

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        max_attempts=1,
        sleep_fn=lambda _s: None,
        enabled=True,
    )
    store = app.extensions["alert_store"]
    app.extensions["notify_config"] = cfg
    app.extensions["notifier"] = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg, opener=opener),
    )
    alert = _make_alert(app)
    _login(client)
    resp = client.post(f"/alerts/{alert.id}/notify", follow_redirects=True)
    assert resp.status_code == 200
    assert store.get(alert.id).delivery_status == "sent"


def test_ack_still_works(app, client):
    alert = _make_alert(app)
    _login(client)
    client.post(
        f"/alerts/{alert.id}/action",
        data={"action": "acknowledge", "note": "Checked feed"},
        follow_redirects=True,
    )
    loaded = app.extensions["alert_store"].get(alert.id)
    assert loaded.status == "acknowledged"
    trail = app.extensions["alert_store"].audit_trail(alert.id)
    assert trail[-1]["note"] == "Checked feed"


def test_alert_detail_hitl_actions_have_no_delete(app, client):
    alert = _make_alert(app)
    _login(client)
    page = client.get(f"/alerts/{alert.id}")
    body = page.get_data(as_text=True)
    assert 'value="acknowledge"' in body
    assert 'value="dismiss"' in body
    assert 'value="escalate"' in body
    assert 'value="reopen"' in body
    assert 'value="delete"' not in body

    resp = client.post(
        f"/alerts/{alert.id}/action",
        data={"action": "delete"},
        follow_redirects=True,
    )
    assert resp.status_code == 200
    assert "Invalid action" in resp.get_data(as_text=True)
    loaded = app.extensions["alert_store"].get(alert.id)
    assert loaded is not None
    assert loaded.status == "open"


def test_cameras_page_has_enable_disable_not_delete(client):
    _login(client)
    body = client.get("/cameras").get_data(as_text=True)
    assert "Enable" in body or "Disable" in body
    assert ">Delete<" not in body
    missing = client.post("/cameras/demo-file-01/delete", follow_redirects=False)
    assert missing.status_code in {404, 405}


def test_relative_snapshot_dir_resolves_and_serves(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.chdir(tmp_path)
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ALERT_DB_PATH": str(tmp_path / "app.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": "data/snapshots",
        }
    )
    resolved = Path(application.config["SNAPSHOT_DIR"])
    assert resolved.is_absolute()
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "frame_demo.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    client = application.test_client()
    client.post("/login", data={"username": "siteadmin", "password": "test-pass-12"})
    resp = client.get("/snapshots/frame_demo.jpg")
    assert resp.status_code == 200
