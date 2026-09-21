"""Mocked Resend email and webhook adapters."""

import io
import json
from urllib.error import HTTPError, URLError

import pytest

from alerts.notify import (
    DEFAULT_FROM,
    DEFAULT_USER_AGENT,
    RESEND_API_URL,
    EmailAdapter,
    NotifyConfig,
    WebhookAdapter,
    combine_delivery_status,
    email_subject,
    merge_recipient_lists,
    merge_recipients,
    parse_env_recipients,
)
from alerts.schema import structured_payload
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


def _payload(store):
    alert = store.create_alert(
        source_label="Authorized Camera — Test",
        category="potential_fall",
        risk_level="elevated",
        confidence=0.71,
        rationale="Possible fall — verify.",
        frame_index=2,
        timestamp_sec=4.0,
        location_label="Gym",
        camera_id="cam-2",
    )
    return structured_payload(alert)


@pytest.fixture
def store(tmp_path):
    return AlertStore(tmp_path / "notify.db")


def test_parse_and_merge_recipients():
    env = parse_env_recipients(" Alpha@x.com, beta@x.com, alpha@x.com, not-an-email ")
    assert env == ["alpha@x.com", "beta@x.com"]
    merged = merge_recipients("alpha@x.com", ["Beta@x.com", "gamma@x.com"])
    assert merged == ["alpha@x.com", "beta@x.com", "gamma@x.com"]
    ordered = merge_recipient_lists(
        ["owner@x.com"],
        ["extra@x.com", "owner@x.com"],
        ["alpha@x.com"],
    )
    assert ordered == ["owner@x.com", "extra@x.com", "alpha@x.com"]


def test_email_without_api_key_is_queued_not_sent(store):
    cfg = NotifyConfig(resend_api_key="", email_recipients_env="op@x.com")
    result = EmailAdapter(cfg).send(_payload(store), ["op@x.com"])
    assert result.status == "queued"
    assert result.ok is False
    assert "RESEND_API_KEY" in result.error


def test_email_without_recipients_is_skipped(store):
    cfg = NotifyConfig(resend_api_key="re_test")
    result = EmailAdapter(cfg).send(_payload(store), [])
    assert result.status == "skipped"
    assert result.ok is False


def test_email_success_posts_resend_with_user_agent(store):
    seen = {}

    def opener(req, timeout=15):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(200, json.dumps({"id": "re_abc"}))

    cfg = NotifyConfig(
        resend_api_key="re_test",
        resend_from=DEFAULT_FROM,
        user_agent=DEFAULT_USER_AGENT,
    )
    result = EmailAdapter(cfg, opener=opener).send(_payload(store), ["op@x.com"])
    assert result.status == "success"
    assert result.provider_id == "re_abc"
    assert seen["url"] == RESEND_API_URL
    assert seen["headers"]["user-agent"] == DEFAULT_USER_AGENT
    assert seen["headers"]["authorization"] == "Bearer re_test"
    assert seen["body"]["from"] == DEFAULT_FROM
    assert seen["body"]["to"] == ["op@x.com"]
    assert "human review" in seen["body"]["subject"].lower()
    assert "AI detects and alerts" in seen["body"]["text"]


def test_email_http_error_is_failed(store):
    def opener(req, timeout=15):
        raise HTTPError(
            req.full_url,
            403,
            "Forbidden",
            hdrs=None,
            fp=io.BytesIO(b"error code: 1010"),
        )

    cfg = NotifyConfig(resend_api_key="re_test")
    result = EmailAdapter(cfg, opener=opener).send(_payload(store), ["op@x.com"])
    assert result.status == "failed"
    assert "403" in result.error
    assert "1010" in result.error


def test_webhook_success_and_auth_header(store):
    seen = {}

    def opener(req, timeout=15):
        seen["url"] = req.full_url
        seen["headers"] = {k.lower(): v for k, v in req.header_items()}
        seen["body"] = json.loads(req.data.decode("utf-8"))
        return _FakeResp(202, '{"ok":true}')

    cfg = NotifyConfig(
        webhook_url="https://siem.example/hooks/alerts",
        webhook_secret="soar-token",
        user_agent=DEFAULT_USER_AGENT,
    )
    result = WebhookAdapter(cfg, opener=opener).send(_payload(store))
    assert result.status == "success"
    assert seen["url"] == "https://siem.example/hooks/alerts"
    assert seen["headers"]["authorization"] == "Bearer soar-token"
    assert seen["headers"]["user-agent"] == DEFAULT_USER_AGENT
    assert seen["body"]["event"] == "mun_cyber_eye.alert"
    assert seen["body"]["alert"]["category"] == "potential_fall"


def test_webhook_missing_url_skipped(store):
    result = WebhookAdapter(NotifyConfig()).send(_payload(store))
    assert result.status == "skipped"


def test_webhook_network_error_failed(store):
    def opener(req, timeout=15):
        raise URLError("connection refused")

    cfg = NotifyConfig(webhook_url="https://siem.example/hook")
    result = WebhookAdapter(cfg, opener=opener).send(_payload(store))
    assert result.status == "failed"
    assert "connection refused" in result.error


def test_combine_delivery_status_never_invents_success():
    from alerts.notify import ChannelResult

    assert (
        combine_delivery_status(
            [
                ChannelResult("email", "queued", error="no key"),
                ChannelResult("webhook", "skipped"),
            ]
        )
        == "queued"
    )
    assert (
        combine_delivery_status(
            [
                ChannelResult("email", "skipped"),
                ChannelResult("webhook", "skipped"),
            ]
        )
        == "undelivered"
    )
    assert (
        combine_delivery_status(
            [
                ChannelResult("email", "success"),
                ChannelResult("webhook", "skipped"),
            ]
        )
        == "sent"
    )
    assert (
        combine_delivery_status(
            [
                ChannelResult("email", "success"),
                ChannelResult("webhook", "failed"),
            ]
        )
        == "partial"
    )
    assert (
        combine_delivery_status(
            [
                ChannelResult("email", "failed"),
                ChannelResult("webhook", "failed"),
            ]
        )
        == "failed"
    )


def test_email_subject_includes_severity(store):
    payload = _payload(store)
    subject = email_subject(payload)
    assert "WARNING" in subject or payload["severity"].upper() in subject
    assert "potential fall" in subject or "potential_fall" in subject
    assert "human review" in subject.lower()
    named = email_subject(
        {
            **payload,
            "camera_name": "Lobby west",
            "place_label": "corridor_hallway",
            "category_label": "potential confrontation",
            "severity": "critical",
        }
    )
    assert "Lobby west" in named
    assert "corridor_hallway" in named
    assert "potential confrontation" in named
