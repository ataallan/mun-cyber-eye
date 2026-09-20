"""Pluggable notification adapters for Phase 4 outbound alerts.

Console/in-app review remains the primary channel. Email (Resend) and an
optional SIEM/SOAR webhook are delivery channels only. Missing credentials
are recorded as queued/undelivered — never reported as success.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

from .schema import SAFETY_BANNER, structured_payload
from .store import Alert, AlertStore

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"
DEFAULT_FROM = "Mun Cyber Technologies <info@muncyber.com>"
DEFAULT_USER_AGENT = "MunCyberEye/4.0 (+https://muncyber.com)"
DEFAULT_MAX_ATTEMPTS = 3


def _http_post_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: float = 15.0,
    opener: Optional[Callable[..., Any]] = None,
) -> tuple[int, str]:
    """POST JSON via urllib. Always send a User-Agent (Cloudflare / Resend)."""
    body = json.dumps(payload).encode("utf-8")
    merged = {
        "Content-Type": "application/json",
        "User-Agent": DEFAULT_USER_AGENT,
        **headers,
    }
    req = urllib.request.Request(url, data=body, headers=merged, method="POST")
    open_fn = opener or urllib.request.urlopen
    try:
        with open_fn(req, timeout=timeout) as resp:
            raw = resp.read()
            text = raw.decode("utf-8", errors="replace") if raw else ""
            status = getattr(resp, "status", None) or resp.getcode()
            return int(status), text
    except urllib.error.HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        return int(exc.code), err_body or str(exc)
    except urllib.error.URLError as exc:
        return 0, str(exc.reason or exc)


@dataclass
class ChannelResult:
    channel: str
    status: str  # success | failed | skipped | undelivered | queued
    recipient: str = ""
    attempt: int = 1
    error: str = ""
    provider_id: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "success"


@dataclass
class NotifyConfig:
    resend_api_key: str = ""
    resend_from: str = DEFAULT_FROM
    email_recipients_env: str = ""
    webhook_url: str = ""
    webhook_secret: str = ""
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    user_agent: str = DEFAULT_USER_AGENT
    enabled: bool = True
    sleep_fn: Callable[[float], None] = field(default=time.sleep)

    @classmethod
    def from_env(cls) -> "NotifyConfig":
        return cls(
            resend_api_key=os.getenv("RESEND_API_KEY", "").strip(),
            resend_from=os.getenv("RESEND_FROM", DEFAULT_FROM).strip() or DEFAULT_FROM,
            email_recipients_env=os.getenv("ALERT_EMAIL_RECIPIENTS", "").strip(),
            webhook_url=os.getenv("ALERT_WEBHOOK_URL", "").strip(),
            webhook_secret=os.getenv("ALERT_WEBHOOK_SECRET", "").strip(),
            max_attempts=max(1, int(os.getenv("ALERT_NOTIFY_MAX_ATTEMPTS", str(DEFAULT_MAX_ATTEMPTS)))),
            user_agent=os.getenv("ALERT_USER_AGENT", DEFAULT_USER_AGENT).strip()
            or DEFAULT_USER_AGENT,
            enabled=os.getenv("ALERT_NOTIFY_ON_CREATE", "1") != "0",
        )

    @property
    def resend_configured(self) -> bool:
        return bool(self.resend_api_key)

    @property
    def webhook_configured(self) -> bool:
        return bool(self.webhook_url)


def parse_env_recipients(raw: str) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for part in (raw or "").split(","):
        email = part.strip().lower()
        if email and "@" in email and email not in seen:
            seen.add(email)
            out.append(email)
    return out


def merge_recipients(env_csv: str, db_emails: Iterable[str]) -> List[str]:
    merged = parse_env_recipients(env_csv)
    seen = set(merged)
    for email in db_emails:
        normalized = (email or "").strip().lower()
        if normalized and "@" in normalized and normalized not in seen:
            seen.add(normalized)
            merged.append(normalized)
    return merged


def email_subject(payload: dict[str, Any]) -> str:
    severity = str(payload.get("severity") or "info").upper()
    category = payload.get("category") or "alert"
    return f"[Mun Cyber Eye] {severity} {category} — human review required"


def email_bodies(payload: dict[str, Any]) -> tuple[str, str]:
    short = payload.get("short_rationale") or payload.get("rationale") or ""
    action = payload.get("recommended_human_action") or ""
    text = (
        f"{SAFETY_BANNER}\n"
        f"Authorized personnel only. No autonomous enforcement.\n\n"
        f"Alert ID: {payload.get('id')}\n"
        f"Correlation ID: {payload.get('correlation_id')}\n"
        f"Created (UTC): {payload.get('created_at')}\n"
        f"Severity: {payload.get('severity')}\n"
        f"Category: {payload.get('category')}\n"
        f"Confidence: {payload.get('confidence')}\n"
        f"Location: {payload.get('location_label') or '—'}\n"
        f"Camera: {payload.get('camera_id') or '—'}\n"
        f"Source: {payload.get('source') or '—'}\n"
        f"Frame time: {payload.get('frame_time')} (#{payload.get('frame_index')})\n"
        f"Human status: {payload.get('human_status')}\n"
        f"Delivery status: {payload.get('delivery_status')}\n\n"
        f"Rationale: {short}\n\n"
        f"Recommended human action (advisory): {action}\n\n"
        f"Open the Mun Cyber Eye console to verify, acknowledge, dismiss, or escalate.\n"
        f"Enforcement: {payload.get('enforcement')}\n"
    )
    html = f"""<!DOCTYPE html>
<html><body style="font-family:Segoe UI,system-ui,sans-serif;background:#0b1220;color:#e8eefc;padding:24px;">
  <div style="background:#1d3b2a;border:1px solid #2f6b4e;padding:12px 16px;border-radius:8px;margin-bottom:16px;">
    <strong>Safety:</strong> {SAFETY_BANNER}<br>
    <span style="color:#a8d7c0;font-size:13px;">Authorized cameras only · No autonomous enforcement</span>
  </div>
  <h1 style="font-size:20px;margin:0 0 8px;">Mun Cyber Eye alert</h1>
  <p style="color:#9aabc8;margin-top:0;">Structured notification for authorized personnel. Humans verify and decide.</p>
  <table style="border-collapse:collapse;width:100%;max-width:640px;">
    <tr><td style="color:#9aabc8;padding:4px 8px;">Severity</td><td>{payload.get("severity")}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Category</td><td>{payload.get("category")}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Confidence</td><td>{payload.get("confidence")}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Location</td><td>{payload.get("location_label") or "—"}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Camera</td><td>{payload.get("camera_id") or "—"}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Source</td><td>{payload.get("source") or "—"}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Frame time</td><td>{payload.get("frame_time")} (#{payload.get("frame_index")})</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Created</td><td>{payload.get("created_at")}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Alert ID</td><td style="font-family:monospace;">{payload.get("id")}</td></tr>
    <tr><td style="color:#9aabc8;padding:4px 8px;">Correlation</td><td style="font-family:monospace;">{payload.get("correlation_id")}</td></tr>
  </table>
  <h2 style="font-size:16px;">Rationale</h2>
  <p>{short}</p>
  <h2 style="font-size:16px;">Recommended human action (advisory only)</h2>
  <p>{action}</p>
  <p style="color:#9aabc8;font-size:13px;">Open the console to acknowledge, dismiss, or escalate. {payload.get("enforcement")}</p>
</body></html>"""
    return html, text


class EmailAdapter:
    """Resend email adapter. Honest about missing keys and HTTP failures."""

    def __init__(
        self,
        config: NotifyConfig,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self.opener = opener

    def send(self, payload: dict[str, Any], recipients: List[str]) -> ChannelResult:
        joined = ", ".join(recipients)
        if not recipients:
            return ChannelResult(
                channel="email",
                status="skipped",
                error="No authorized email recipients configured",
            )
        if not self.config.resend_api_key:
            return ChannelResult(
                channel="email",
                status="queued",
                recipient=joined,
                error="RESEND_API_KEY not configured; email not sent",
            )

        html, text = email_bodies(payload)
        request_payload = {
            "from": self.config.resend_from,
            "to": recipients,
            "subject": email_subject(payload),
            "html": html,
            "text": text,
        }
        headers = {
            "Authorization": f"Bearer {self.config.resend_api_key}",
            "User-Agent": self.config.user_agent,
        }
        status, body = _http_post_json(
            RESEND_API_URL,
            request_payload,
            headers,
            opener=self.opener,
        )
        provider_id = ""
        try:
            parsed = json.loads(body) if body else {}
            if isinstance(parsed, dict):
                provider_id = str(parsed.get("id") or "")
        except json.JSONDecodeError:
            parsed = {}

        if 200 <= status < 300:
            return ChannelResult(
                channel="email",
                status="success",
                recipient=joined,
                provider_id=provider_id,
            )
        return ChannelResult(
            channel="email",
            status="failed",
            recipient=joined,
            error=f"Resend HTTP {status}: {body[:500] or 'no response body'}",
        )


class WebhookAdapter:
    """Optional SIEM/SOAR webhook. POSTs the structured alert JSON."""

    def __init__(
        self,
        config: NotifyConfig,
        opener: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.config = config
        self.opener = opener

    def send(self, payload: dict[str, Any]) -> ChannelResult:
        url = self.config.webhook_url
        if not url:
            return ChannelResult(
                channel="webhook",
                status="skipped",
                error="ALERT_WEBHOOK_URL not configured",
            )
        envelope = {
            "event": "mun_cyber_eye.alert",
            "safety": SAFETY_BANNER,
            "alert": payload,
        }
        headers = {"User-Agent": self.config.user_agent}
        if self.config.webhook_secret:
            headers["Authorization"] = f"Bearer {self.config.webhook_secret}"
        status, body = _http_post_json(url, envelope, headers, opener=self.opener)
        if 200 <= status < 300:
            return ChannelResult(
                channel="webhook",
                status="success",
                recipient=url,
                provider_id=str(status),
            )
        return ChannelResult(
            channel="webhook",
            status="failed",
            recipient=url,
            error=f"Webhook HTTP {status}: {body[:500] or 'no response body'}",
        )


def combine_delivery_status(results: List[ChannelResult]) -> str:
    """Map channel outcomes to an honest alert-level delivery_status."""
    actionable = [r for r in results if r.status != "skipped"]
    if not actionable:
        return "undelivered"

    statuses = {r.status for r in actionable}
    successes = {r.status for r in actionable if r.status == "success"}
    failures = {r.status for r in actionable if r.status == "failed"}
    queued = {r.status for r in actionable if r.status in {"queued", "undelivered"}}

    if successes and not failures and not queued:
        return "sent"
    if successes and (failures or queued):
        return "partial"
    if failures and not successes and not queued:
        return "failed"
    if queued and not successes and not failures:
        # Missing key / not actually sent — never invent success.
        return "queued" if any(r.status == "queued" for r in actionable) else "undelivered"
    if failures and queued:
        return "failed"
    return "undelivered"


class NotificationService:
    """Deliver an alert over configured adapters and persist attempt logs."""

    def __init__(
        self,
        store: AlertStore,
        config: Optional[NotifyConfig] = None,
        email_adapter: Optional[EmailAdapter] = None,
        webhook_adapter: Optional[WebhookAdapter] = None,
    ) -> None:
        self.store = store
        self.config = config or NotifyConfig.from_env()
        self.email = email_adapter or EmailAdapter(self.config)
        self.webhook = webhook_adapter or WebhookAdapter(self.config)

    def recipient_emails(self) -> List[str]:
        db_emails = [r["email"] for r in self.store.list_recipients(active_only=True)]
        return merge_recipients(self.config.email_recipients_env, db_emails)

    def deliver(
        self,
        alert: Alert,
        *,
        actor: str = "system",
        force: bool = False,
    ) -> Alert:
        if not self.config.enabled and not force:
            self.store.set_delivery_status(alert.id, "undelivered")
            self.store.record_audit(
                alert.id,
                "notify_skipped",
                actor,
                "Outbound notify disabled (ALERT_NOTIFY_ON_CREATE=0)",
            )
            return self.store.get(alert.id) or alert

        payload = structured_payload(alert)
        results = self._run_channels(payload)
        status = combine_delivery_status(results)
        self.store.set_delivery_status(alert.id, status)
        for result in results:
            self.store.log_delivery(
                alert.id,
                channel=result.channel,
                recipient=result.recipient,
                status=result.status,
                attempt=result.attempt,
                error=result.error,
                provider_id=result.provider_id,
            )
        summary = "; ".join(
            f"{r.channel}={r.status}" + (f" ({r.error})" if r.error and r.status != "success" else "")
            for r in results
        )
        self.store.record_audit(alert.id, "notify" if actor == "system" else "resend", actor, summary)
        updated = self.store.get(alert.id) or alert
        logger.info("Alert %s delivery_status=%s %s", alert.id[:8], status, summary)
        return updated

    def _run_channels(self, payload: dict[str, Any]) -> List[ChannelResult]:
        results: List[ChannelResult] = []
        results.append(self._with_retries(lambda: self.email.send(payload, self.recipient_emails()), "email"))
        results.append(self._with_retries(lambda: self.webhook.send(payload), "webhook"))
        return results

    def _with_retries(self, send_fn: Callable[[], ChannelResult], channel: str) -> ChannelResult:
        last = ChannelResult(channel=channel, status="failed", error="no attempt")
        for attempt in range(1, self.config.max_attempts + 1):
            last = send_fn()
            last.attempt = attempt
            if last.status in {"success", "skipped", "queued", "undelivered"}:
                return last
            if attempt < self.config.max_attempts:
                self.config.sleep_fn(min(2.0, 0.25 * attempt))
        return last


def send_resend_email(
    *,
    api_key: str,
    from_addr: str,
    to: str,
    subject: str,
    html: str,
    text: Optional[str] = None,
    timeout: float = 20,
    user_agent: str = DEFAULT_USER_AGENT,
    opener: Optional[Callable[..., Any]] = None,
) -> tuple[bool, str]:
    """Send one transactional email (password reset). Honest if the key is missing."""
    if not (api_key or "").strip():
        return False, "RESEND_API_KEY is not configured"
    if not (from_addr or "").strip():
        return False, "RESEND_FROM is not configured"
    if not (to or "").strip():
        return False, "Recipient is missing"

    payload: dict[str, Any] = {
        "from": from_addr.strip(),
        "to": [to.strip()],
        "subject": subject,
        "html": html,
    }
    if text:
        payload["text"] = text

    status, body = _http_post_json(
        RESEND_API_URL,
        payload,
        {
            "Authorization": f"Bearer {api_key.strip()}",
            "User-Agent": (user_agent or DEFAULT_USER_AGENT),
        },
        timeout=timeout,
        opener=opener,
    )
    provider_id = ""
    try:
        parsed = json.loads(body) if body else {}
        if isinstance(parsed, dict):
            provider_id = str(parsed.get("id") or "").strip()
    except json.JSONDecodeError:
        provider_id = ""
    if 200 <= status < 300:
        return True, provider_id or f"http-{status}"
    if status == 0:
        return False, f"Resend request failed: {body[:240] or 'no response'}"
    return False, f"Resend HTTP {status}: {body[:240] or 'no response body'}"
