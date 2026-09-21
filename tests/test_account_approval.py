"""Pending customer registrations must be approved before console access."""

from __future__ import annotations

import sqlite3

from app.auth import PENDING_LOGIN_MESSAGE, UserStore
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
        "ALERT_NOTIFY_ON_CREATE": False,
        "SEED_DEMO_CAMERAS": False,
    }
    config.update(extra)
    return create_app(config)


def _register(client, username, email, password="test-pass-12"):
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


def _login(client, username, password="test-pass-12", follow=True):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=follow,
    )


def test_first_user_auto_approved_can_login(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    created = _register(client, "founder", "founder@example.com")
    assert created.status_code == 200
    store: UserStore = app.extensions["user_store"]
    founder = store.get_by_username("founder")
    assert founder is not None
    assert founder.role == "admin"
    assert founder.approved is True
    assert founder.approved_by == "bootstrap"
    ok = _login(client, "founder")
    assert ok.status_code == 200
    body = ok.get_data(as_text=True)
    assert "Alert console" in body
    assert "Accounts" in body


def test_pending_login_blocked_until_approve(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    pending_page = _register(client, "reviewer", "reviewer@example.com")
    assert "pending admin approval" in pending_page.get_data(as_text=True).lower()

    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    assert reviewer is not None
    assert reviewer.role == "operator"
    assert reviewer.approved is False
    assert reviewer.access_status() == "pending"

    blocked = _login(client, "reviewer")
    body = blocked.get_data(as_text=True)
    assert blocked.status_code == 200
    assert PENDING_LOGIN_MESSAGE in body
    assert "Alert console" not in body

    wrong = _login(client, "reviewer", password="wrong-pass")
    assert "Invalid credentials" in wrong.get_data(as_text=True)
    assert "awaiting admin approval" not in wrong.get_data(as_text=True).lower()

    _login(client, "founder")
    accounts = client.get("/accounts")
    assert accounts.status_code == 200
    acc_body = accounts.get_data(as_text=True)
    assert "reviewer" in acc_body
    assert "pending" in acc_body.lower()
    assert "Approve" in acc_body

    approve = client.post(
        f"/accounts/{reviewer.id}/approve",
        follow_redirects=True,
    )
    assert approve.status_code == 200
    assert "Approved reviewer" in approve.get_data(as_text=True)
    refreshed = store.get_by_username("reviewer")
    assert refreshed is not None
    assert refreshed.approved is True
    assert refreshed.approved_by == "founder"
    actions = [row["action"] for row in app.extensions["alert_store"].list_system_audit()]
    assert "approve_account" in actions

    client.get("/logout", follow_redirects=True)
    ok = _login(client, "reviewer")
    assert "Alert console" in ok.get_data(as_text=True)
    assert "reviewer" in ok.get_data(as_text=True)


def test_env_seeded_accounts_are_approved(tmp_path):
    app = _fresh_app(
        tmp_path,
        ADMIN_USERNAME="siteadmin",
        ADMIN_PASSWORD="test-pass-12",
        ADMIN_EMAIL="siteadmin@localhost",
        ADMIN_SYNC_PASSWORD=True,
        ADMIN_ROLE="admin",
        DEVELOPER_USERNAME="muncyber",
        DEVELOPER_PASSWORD="lab-secret-99",
        DEVELOPER_EMAIL="lab@muncyber.example",
    )
    store: UserStore = app.extensions["user_store"]
    admin = store.get_by_username("siteadmin")
    dev = store.get_by_username("muncyber")
    assert admin is not None and admin.approved is True and admin.role == "admin"
    assert admin.approved_by == "env"
    assert dev is not None and dev.approved is True and dev.role == "developer"
    assert dev.approved_by == "env"
    client = app.test_client()
    admin_ok = _login(client, "siteadmin", "test-pass-12")
    assert "Alert console" in admin_ok.get_data(as_text=True)
    client.get("/logout", follow_redirects=True)
    dev_ok = _login(client, "muncyber", "lab-secret-99")
    assert "Alert console" in dev_ok.get_data(as_text=True)
    assert ">Accounts</a>" in dev_ok.get_data(as_text=True)


def test_operator_cannot_approve(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    _register(client, "reviewer", "reviewer@example.com")
    _register(client, "later", "later@example.com")
    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    later = store.get_by_username("later")
    assert reviewer is not None and later is not None
    store.approve_user(reviewer.id, "founder")

    _login(client, "reviewer")
    page = client.get("/accounts", follow_redirects=True)
    body = page.get_data(as_text=True)
    assert "Only a site admin or Mun Cyber developer can approve accounts." in body
    assert ">Accounts</a>" not in client.get("/").get_data(as_text=True)

    denied = client.post(
        f"/accounts/{later.id}/approve",
        follow_redirects=True,
    )
    assert "Only a site admin or Mun Cyber developer can approve accounts." in denied.get_data(
        as_text=True
    )
    still = store.get_by_username("later")
    assert still is not None
    assert still.approved is False


def test_developer_can_approve_pending_customer(tmp_path):
    app = _fresh_app(
        tmp_path,
        DEVELOPER_USERNAME="muncyber",
        DEVELOPER_PASSWORD="lab-secret-99",
        DEVELOPER_EMAIL="lab@muncyber.example",
        ADMIN_SYNC_PASSWORD=True,
    )
    client = app.test_client()
    created = _register(client, "customer", "customer@example.com")
    assert "pending admin approval" in created.get_data(as_text=True).lower()
    store: UserStore = app.extensions["user_store"]
    customer = store.get_by_username("customer")
    assert customer is not None
    assert customer.role == "operator"
    assert customer.approved is False

    _login(client, "muncyber", "lab-secret-99")
    page = client.get("/accounts")
    assert page.status_code == 200
    assert "customer" in page.get_data(as_text=True)
    client.post(f"/accounts/{customer.id}/approve", follow_redirects=True)
    refreshed = store.get_by_username("customer")
    assert refreshed is not None
    assert refreshed.approved is True
    assert refreshed.approved_by == "muncyber"
    client.get("/logout", follow_redirects=True)
    ok = _login(client, "customer")
    assert "Alert console" in ok.get_data(as_text=True)


def test_pending_password_reset_does_not_grant_access(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    _register(client, "reviewer", "reviewer@example.com", password="old-pass-12x")
    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    assert reviewer is not None
    assert reviewer.approved is False

    forgot = client.post(
        "/forgot-password",
        data={"identifier": "reviewer@example.com"},
        follow_redirects=True,
    )
    body = forgot.get_data(as_text=True)
    assert "reset-password?token=" not in body
    assert store.get_by_username("reviewer").reset_token is None

    token = store.create_reset_token(reviewer.id, ttl_minutes=45)
    reset_get = client.get(f"/reset-password?token={token}", follow_redirects=True)
    assert "invalid or has expired" in reset_get.get_data(as_text=True)

    store.approve_user(reviewer.id, "founder")
    still = _login(client, "reviewer", "old-pass-12x")
    assert "Alert console" in still.get_data(as_text=True)


def test_reject_then_cannot_login(tmp_path):
    app = _fresh_app(tmp_path)
    client = app.test_client()
    _register(client, "founder", "founder@example.com")
    _register(client, "reviewer", "reviewer@example.com")
    store: UserStore = app.extensions["user_store"]
    reviewer = store.get_by_username("reviewer")
    assert reviewer is not None
    _login(client, "founder")
    rejected = client.post(
        f"/accounts/{reviewer.id}/reject",
        follow_redirects=True,
    )
    assert "Rejected reviewer" in rejected.get_data(as_text=True)
    actions = [row["action"] for row in app.extensions["alert_store"].list_system_audit()]
    assert "reject_account" in actions
    client.get("/logout", follow_redirects=True)
    blocked = _login(client, "reviewer")
    assert "inactive" in blocked.get_data(as_text=True).lower()
    assert "Alert console" not in blocked.get_data(as_text=True)


def test_legacy_users_remain_approved(tmp_path):
    db = tmp_path / "legacy-auth.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            email TEXT NOT NULL COLLATE NOCASE UNIQUE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('admin', 'operator')),
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            reset_token TEXT,
            reset_expires TEXT
        );
        INSERT INTO users (
            id, username, email, password_hash, role, active, created_at
        ) VALUES (
            'legacy-1', 'oldadmin', 'oldadmin@localhost',
            'not-a-real-hash', 'admin', 1, '2026-01-01T00:00:00Z'
        );
        """
    )
    conn.commit()
    conn.close()
    store = UserStore(db)
    user = store.get_by_username("oldadmin")
    assert user is not None
    assert user.approved is True
    assert user.approved_by == "legacy"
    assert user.can_access_console()
