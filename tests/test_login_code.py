"""Email sign-in codes after approval — customer 2FA, not TOTP."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from app.auth import (
    LOGIN_CODE_MAX_ATTEMPTS,
    PENDING_LOGIN_MESSAGE,
    LoginCodeCooldown,
    UserStore,
)
from app.factory import create_app


def _fresh_app(tmp_path, **extra):
    config = {
        "TESTING": True,
        "SECRET_KEY": "test-secret",
        "ADMIN_USERNAME": "",
        "ADMIN_PASSWORD": "",
        "ADMIN_EMAIL": "",
        "ADMIN_SYNC_PASSWORD": False,
        "DEVELOPER_USERNAME": "",
        "DEVELOPER_PASSWORD": "",
        "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
        "AUTH_DB_PATH": str(tmp_path / "auth.db"),
        "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        "RESEND_API_KEY": "",
        "RESEND_FROM": "",
        "AUTH_SHOW_RESET_URL": True,
        "AUTH_SHOW_LOGIN_CODE": True,
        "ALERT_NOTIFY_ON_CREATE": False,
        "SEED_DEMO_CAMERAS": False,
        "CUSTOMER_2FA_REQUIRED": True,
        "DEVELOPER_2FA_REQUIRED": False,
    }
    config.update(extra)
    return create_app(config)


def _register(client, username, email, password="secret123"):
    return client.post(
        "/register",
        data={
            "username": username,
            "email": email,
            "password": password,
            "confirm_password": password,
        },
        follow_redirects=True,
    )


def _login(client, username, password="secret123", follow=True):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=follow,
    )


def _demo_code(html: str) -> str:
    match = re.search(r'class="mono auth-code-demo">(\d{6})</p>', html)
    assert match, html
    return match.group(1)


def test_pending_login_blocked_without_2fa_path(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    _register(client, "reviewer", "reviewer@example.com")
    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    assert reviewer is not None
    assert reviewer.approved is False
    assert reviewer.login_code_hash is None

    with patch("app.auth.generate_login_code") as mocked:
        blocked = _login(client, "reviewer")
    mocked.assert_not_called()
    body = blocked.get_data(as_text=True)
    assert PENDING_LOGIN_MESSAGE in body
    assert "Alert console" not in body
    assert "Enter sign-in code" not in body
    assert "was not emailed" not in body
    refreshed = store.get_by_username("reviewer")
    assert refreshed is not None
    assert refreshed.login_code_hash is None

    forged = client.get("/login-code", follow_redirects=True)
    assert "Sign in with your password first" in forged.get_data(as_text=True)

    with client.session_transaction() as sess:
        sess["pending_2fa_user"] = "reviewer"
    sneak = client.get("/login-code", follow_redirects=True)
    sneak_body = sneak.get_data(as_text=True)
    assert PENDING_LOGIN_MESSAGE in sneak_body
    assert "Enter sign-in code" not in sneak_body
    still = store.get_by_username("reviewer")
    assert still is not None
    assert still.login_code_hash is None


def test_approved_customer_must_enter_email_code(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    _register(client, "reviewer", "reviewer@example.com")
    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    assert reviewer is not None
    store.approve_user(reviewer.id, "founder")

    challenge = _login(client, "reviewer")
    body = challenge.get_data(as_text=True)
    assert challenge.status_code == 200
    assert "Enter sign-in code" in body
    assert "Alert console" not in body
    assert "Email delivery is not configured" in body
    assert "A sign-in code was not emailed" in body
    assert "A sign-in code was sent to the login email" not in body
    assert "reviewer@example.com" in body
    code = _demo_code(body)

    hashed = store.get_by_username("reviewer")
    assert hashed is not None
    assert hashed.login_code_hash
    assert code not in hashed.login_code_hash
    assert hashed.login_code_hash != code

    dash = client.get("/", follow_redirects=True)
    assert "Enter sign-in code" in dash.get_data(as_text=True)
    assert "Alert console" not in dash.get_data(as_text=True)

    wrong = client.post("/login-code", data={"code": "000000"}, follow_redirects=True)
    assert "incorrect" in wrong.get_data(as_text=True).lower()
    assert "Alert console" not in wrong.get_data(as_text=True)

    ok = client.post("/login-code", data={"code": code}, follow_redirects=True)
    assert "Alert console" in ok.get_data(as_text=True)
    assert "reviewer" in ok.get_data(as_text=True)
    consumed = store.get_by_username("reviewer")
    assert consumed is not None
    assert consumed.login_code_hash is None


def test_first_admin_email_code_then_console(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    challenge = _login(client, "founder")
    body = challenge.get_data(as_text=True)
    assert "Enter sign-in code" in body
    assert "Alert console" not in body
    code = _demo_code(body)
    ok = client.post("/login-code", data={"code": code}, follow_redirects=True)
    assert "Alert console" in ok.get_data(as_text=True)
    assert "Accounts" in ok.get_data(as_text=True)


def test_developer_skips_email_2fa_by_default(tmp_path):
    app = _fresh_app(
        tmp_path,
        DEVELOPER_USERNAME="muncyber",
        DEVELOPER_PASSWORD="lab-secret-99",
        DEVELOPER_EMAIL="lab@muncyber.example",
        ADMIN_SYNC_PASSWORD=True,
        CUSTOMER_2FA_REQUIRED=True,
        DEVELOPER_2FA_REQUIRED=False,
    )
    client = app.test_client()
    ok = _login(client, "muncyber", "lab-secret-99")
    body = ok.get_data(as_text=True)
    assert "Alert console" in body
    assert "Enter sign-in code" not in body
    assert ">Train models</a>" in body
    store: UserStore = app.extensions["user_store"]
    dev = store.get_by_username("muncyber")
    assert dev is not None
    assert dev.login_code_hash is None


def test_customer_2fa_off_skips_code(tmp_path):
    app = _fresh_app(tmp_path, CUSTOMER_2FA_REQUIRED=False)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    ok = _login(client, "founder")
    assert "Alert console" in ok.get_data(as_text=True)
    assert "Enter sign-in code" not in ok.get_data(as_text=True)


def test_login_code_expired_and_single_use(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    store: UserStore = app.extensions["user_store"]
    founder = store.get_by_username("founder")
    assert founder is not None

    challenge = _login(client, "founder")
    code = _demo_code(challenge.get_data(as_text=True))
    expired = (datetime.now(timezone.utc) - timedelta(minutes=1)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with store._conn() as conn:
        conn.execute(
            "UPDATE users SET login_code_expires = ? WHERE id = ?",
            (expired, founder.id),
        )
    stale = client.post("/login-code", data={"code": code}, follow_redirects=True)
    assert "invalid or has expired" in stale.get_data(as_text=True)
    assert "Alert console" not in stale.get_data(as_text=True)

    again = _login(client, "founder")
    fresh = _demo_code(again.get_data(as_text=True))
    first = client.post("/login-code", data={"code": fresh}, follow_redirects=True)
    assert "Alert console" in first.get_data(as_text=True)
    client.get("/logout", follow_redirects=True)
    replay_login = _login(client, "founder")
    replay = client.post(
        "/login-code", data={"code": fresh}, follow_redirects=True
    )
    body = replay.get_data(as_text=True)
    login_body = replay_login.get_data(as_text=True)
    assert "Enter sign-in code" in login_body
    assert "Alert console" not in body
    assert "incorrect" in body.lower() or "invalid" in body.lower()


def test_login_code_rate_limit_and_max_attempts(tmp_path):
    store = UserStore(tmp_path / "auth.db")
    user = store.create_user(
        username="siteadmin",
        email="siteadmin@example.com",
        password="secret123",
        role="admin",
        approved=True,
        approved_by="test",
    )
    first = store.create_login_code(user.id, min_resend_seconds=45)
    assert len(first) == 6
    try:
        store.create_login_code(user.id, min_resend_seconds=45)
        raise AssertionError("expected cooldown")
    except LoginCodeCooldown as exc:
        assert exc.seconds >= 1

    pending = store.create_user(
        username="waiting",
        email="waiting@example.com",
        password="secret123",
        role="operator",
        approved=False,
    )
    try:
        store.create_login_code(pending.id)
        raise AssertionError("pending users must not get codes")
    except ValueError as exc:
        assert "approved" in str(exc).lower()
    reloaded = store.get_by_id(pending.id)
    assert reloaded is not None
    assert reloaded.login_code_hash is None

    target = store.create_user(
        username="lockout",
        email="lockout@example.com",
        password="secret123",
        role="operator",
        approved=True,
        approved_by="test",
    )
    real = store.create_login_code(target.id, force=True)
    for _ in range(LOGIN_CODE_MAX_ATTEMPTS - 1):
        try:
            store.verify_login_code(target.id, "000000")
        except ValueError as exc:
            assert "incorrect" in str(exc).lower()
    try:
        store.verify_login_code(target.id, "000000")
        raise AssertionError("expected lockout")
    except ValueError as exc:
        assert "too many" in str(exc).lower()
    locked = store.get_by_id(target.id)
    assert locked is not None
    assert locked.login_code_hash is None
    try:
        store.verify_login_code(target.id, real)
        raise AssertionError("cleared code must not verify")
    except ValueError:
        pass


def test_resend_configured_claims_email_only_on_success(tmp_path):
    app = _fresh_app(
        tmp_path,
        RESEND_API_KEY="re_test",
        RESEND_FROM="Mun Cyber Eye <ops@example.com>",
        LOGIN_CODE_RESEND_SECONDS=0,
    )
    client = app.test_client()
    _register(client, "founder", "founder@example.com")

    with patch("app.routes.send_resend_email", return_value=(True, "re_msg")) as sent:
        ok = _login(client, "founder")
    sent.assert_called_once()
    kwargs = sent.call_args.kwargs
    assert kwargs["to"] == "founder@example.com"
    body = ok.get_data(as_text=True)
    assert "A sign-in code was sent to the login email" in body
    assert "A sign-in code was not emailed" not in body
    code = _demo_code(body)
    assert code in kwargs["text"]

    client.get("/logout", follow_redirects=True)
    with patch("app.routes.send_resend_email", return_value=(False, "provider down")):
        failed = _login(client, "founder")
    fail_body = failed.get_data(as_text=True)
    assert "Email delivery failed" in fail_body
    assert "A sign-in code was not emailed" in fail_body
    assert "A sign-in code was sent to the login email" not in fail_body
    assert "Enter sign-in code" in fail_body
