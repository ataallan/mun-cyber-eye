"""Delivery status transitions, retries, and recipient directory."""

import json

import pytest

from alerts.notify import EmailAdapter, NotificationService, NotifyConfig, WebhookAdapter
from alerts.store import AlertStore


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
def store(tmp_path):
    return AlertStore(tmp_path / "delivery.db")


def _alert(store, **kwargs):
    defaults = dict(
        source_label="Cam",
        category="potential_fight",
        risk_level="high",
        confidence=0.8,
        rationale="Possible fight — verify.",
        frame_index=1,
        timestamp_sec=1.0,
    )
    defaults.update(kwargs)
    return store.create_alert(**defaults)


def test_no_channels_marks_undelivered(store):
    alert = _alert(store)
    svc = NotificationService(
        store,
        NotifyConfig(resend_api_key="", email_recipients_env="", webhook_url=""),
    )
    updated = svc.deliver(alert, actor="system")
    assert updated.delivery_status == "undelivered"
    trail = store.delivery_trail(alert.id)
    assert {row["channel"] for row in trail} == {"email", "webhook"}
    assert all(row["status"] == "skipped" for row in trail)
    audit = [e["action"] for e in store.audit_trail(alert.id)]
    assert "created" in audit
    assert "notify" in audit


def test_missing_resend_key_with_recipients_is_queued(store):
    store.add_recipient("oncall@example.com", created_by="admin")
    alert = _alert(store)
    svc = NotificationService(
        store,
        NotifyConfig(resend_api_key="", email_recipients_env=""),
    )
    updated = svc.deliver(alert)
    assert updated.delivery_status == "queued"
    email_rows = [r for r in store.delivery_trail(alert.id) if r["channel"] == "email"]
    assert email_rows[-1]["status"] == "queued"
    assert "RESEND_API_KEY" in email_rows[-1]["error"]


def test_email_success_sets_sent(store):
    def opener(req, timeout=15):
        return _FakeResp(200, json.dumps({"id": "re_ok"}))

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        max_attempts=1,
        sleep_fn=lambda _s: None,
    )
    svc = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg, opener=opener),
    )
    alert = _alert(store)
    updated = svc.deliver(alert)
    assert updated.delivery_status == "sent"
    assert store.get(alert.id).delivery_status == "sent"


def test_email_fail_webhook_success_is_partial(store):
    def opener(req, timeout=15):
        if "resend.com" in req.full_url:
            return _FakeResp(500, '{"error":"nope"}')
        return _FakeResp(200, '{"ok":true}')

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        webhook_url="https://siem.example/hook",
        max_attempts=1,
        sleep_fn=lambda _s: None,
    )
    svc = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg, opener=opener),
    )
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "partial"


def test_retries_then_fails(store):
    attempts = {"n": 0}

    def opener(req, timeout=15):
        attempts["n"] += 1
        return _FakeResp(503, "busy")

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        max_attempts=3,
        sleep_fn=lambda _s: None,
    )
    svc = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(NotifyConfig(), opener=opener),
    )
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "failed"
    assert attempts["n"] == 3
    email_rows = [r for r in store.delivery_trail(updated.id) if r["channel"] == "email"]
    assert email_rows[-1]["attempt"] == 3
    assert email_rows[-1]["status"] == "failed"


def test_disabled_notify_is_undelivered_not_sent(store):
    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        enabled=False,
    )
    svc = NotificationService(store, cfg)
    updated = svc.deliver(_alert(store))
    assert updated.delivery_status == "undelivered"
    assert "notify_skipped" in [e["action"] for e in store.audit_trail(updated.id)]


def test_resend_appends_audit_and_delivery(store):
    def opener(req, timeout=15):
        return _FakeResp(200, json.dumps({"id": "re_2"}))

    cfg = NotifyConfig(
        resend_api_key="re_test",
        email_recipients_env="op@example.com",
        max_attempts=1,
        sleep_fn=lambda _s: None,
    )
    svc = NotificationService(
        store,
        cfg,
        email_adapter=EmailAdapter(cfg, opener=opener),
        webhook_adapter=WebhookAdapter(cfg),
    )
    alert = _alert(store)
    svc.deliver(alert, actor="system")
    svc.deliver(alert, actor="operator", force=True)
    actions = [e["action"] for e in store.audit_trail(alert.id)]
    assert actions.count("notify") == 1
    assert actions.count("resend") == 1
    assert store.get(alert.id).status == "open"


def test_ack_and_delivery_remain_independent(store):
    alert = _alert(store)
    store.set_delivery_status(alert.id, "queued")
    store.apply_action(alert.id, "acknowledge", actor="operator", note="Verified")
    loaded = store.get(alert.id)
    assert loaded.status == "acknowledged"
    assert loaded.delivery_status == "queued"
    trail = store.audit_trail(alert.id)
    assert trail[-1]["action"] == "acknowledge"


def test_recipient_crud_and_active_filter(store):
    store.add_recipient("a@x.com", display_name="A", role="operator", created_by="admin")
    row = store.add_recipient("b@x.com", display_name="B", created_by="admin")
    store.set_recipient_active(row["id"], False)
    active = store.list_recipients(active_only=True)
    assert [r["email"] for r in active] == ["a@x.com"]
    with pytest.raises(ValueError):
        store.add_recipient("not-valid")
    with pytest.raises(ValueError):
        store.add_recipient("ok@x.com", role="viewer")


def test_reactivate_existing_recipient(store):
    row = store.add_recipient("op@x.com", created_by="admin")
    store.set_recipient_active(row["id"], False)
    again = store.add_recipient("op@x.com", display_name="On call", created_by="admin")
    assert again["active"] == 1
    assert again["display_name"] == "On call"


def test_invalid_delivery_status_rejected(store):
    alert = _alert(store)
    with pytest.raises(ValueError):
        store.set_delivery_status(alert.id, "delivered_probably")
