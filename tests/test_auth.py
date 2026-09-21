"""Console registration, login, and password reset."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from alerts.notify import DEFAULT_USER_AGENT, send_resend_email
from app.factory import create_app
from app.auth import UserStore, public_register_role


@pytest.fixture
def app(tmp_path):
    """Fresh install: no env bootstrap, no default password."""
    return create_app(
        {
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
        }
    )


@pytest.fixture
def client(app):
    return app.test_client()


def _register(client, username="alice", email="alice@example.com", password="secret123"):
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


def test_login_page_has_logo_and_create_account(client):
    rv = client.get("/login")
    assert rv.status_code == 200
    body = rv.get_data(as_text=True)
    assert "mun-cyber-eye-logo.png" in body
    assert "Don't have an account?" in body
    assert "Create account" in body
    assert "Forgot password?" in body
    assert "Create the first admin account" in body
    assert "changeme" not in body
    assert "default password" in body.lower()


def test_fresh_install_has_no_default_operator_changeme(app, client):
    store: UserStore = app.extensions["user_store"]
    assert store.count() == 0
    assert store.get_by_username("operator") is None
    rv = client.post(
        "/login",
        data={"username": "operator", "password": "changeme"},
    )
    assert rv.status_code == 200
    assert "Invalid credentials" in rv.get_data(as_text=True)
    assert "Alert console" not in rv.get_data(as_text=True)


def test_register_then_login(client):
    rv = _register(client)
    assert rv.status_code == 200
    assert "Account created" in rv.get_data(as_text=True) or "site admin" in rv.get_data(
        as_text=True
    )

    bad = client.post(
        "/login",
        data={"username": "alice", "password": "wrong-password"},
    )
    assert bad.status_code == 200
    assert "Invalid credentials" in bad.get_data(as_text=True)

    ok = client.post(
        "/login",
        data={"username": "alice", "password": "secret123"},
        follow_redirects=True,
    )
    assert ok.status_code == 200
    assert "Alert console" in ok.get_data(as_text=True)
    assert "alice" in ok.get_data(as_text=True)


def test_first_register_is_admin_second_is_operator(app, client):
    first = _register(client, username="founder", email="founder@example.com")
    assert first.status_code == 200
    assert "site admin" in first.get_data(as_text=True).lower()
    founder = app.extensions["user_store"].get_by_username("founder")
    assert founder is not None
    assert founder.role == "admin"
    assert public_register_role(is_first=True) == "admin"
    assert public_register_role(is_first=False) == "operator"

    second = _register(client, username="reviewer", email="reviewer@example.com")
    assert second.status_code == 200
    reviewer = app.extensions["user_store"].get_by_username("reviewer")
    assert reviewer is not None
    assert reviewer.role == "operator"
    assert reviewer.approved is False
    assert "pending admin approval" in second.get_data(as_text=True).lower()


def test_register_ignores_posted_developer_role(app, client):
    rv = client.post(
        "/register",
        data={
            "username": "sneaky",
            "email": "sneaky@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "role": "developer",
        },
        follow_redirects=True,
    )
    assert rv.status_code == 200
    user = app.extensions["user_store"].get_by_username("sneaky")
    assert user is not None
    assert user.role == "admin"


def test_optional_env_bootstrap_when_both_set(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "siteadmin@localhost",
            "ADMIN_SYNC_PASSWORD": True,
            "ADMIN_ROLE": "admin",
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        }
    )
    client = application.test_client()
    bad = client.post(
        "/login",
        data={"username": "siteadmin", "password": "not-the-password"},
    )
    assert "Invalid credentials" in bad.get_data(as_text=True)
    ok = client.post(
        "/login",
        data={"username": "siteadmin", "password": "test-pass-12"},
        follow_redirects=True,
    )
    assert ok.status_code == 200
    text = ok.get_data(as_text=True)
    assert "Alert console" in text
    assert "siteadmin" in text
    assert "admin" in text


def test_register_validation(client):
    rv = client.post(
        "/register",
        data={
            "username": "ab",
            "email": "not-an-email",
            "password": "short",
            "confirm_password": "different",
        },
    )
    assert rv.status_code == 200
    assert "Username must be" in rv.get_data(as_text=True)

    _register(client, username="bob", email="bob@example.com")
    again = client.post(
        "/register",
        data={
            "username": "bob",
            "email": "other@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
        },
    )
    assert "already taken" in again.get_data(as_text=True)


def test_forgot_password_token_reset(client, app):
    _register(client, username="casey", email="casey@example.com", password="oldpass12")
    rv = client.post(
        "/forgot-password",
        data={"identifier": "casey@example.com"},
        follow_redirects=True,
    )
    body = rv.get_data(as_text=True)
    assert "Email delivery is not configured" in body
    assert "email sent" not in body.lower()
    assert "reset-password?token=" in body

    with app.app_context():
        user = app.extensions["user_store"].get_by_username("casey")
        assert user is not None
        token = user.reset_token
        assert token

    reset = client.post(
        "/reset-password",
        data={
            "token": token,
            "password": "newpass99",
            "confirm_password": "newpass99",
        },
        follow_redirects=True,
    )
    assert "Password updated" in reset.get_data(as_text=True)

    still_old = client.post(
        "/login",
        data={"username": "casey", "password": "oldpass12"},
    )
    assert "Invalid credentials" in still_old.get_data(as_text=True)

    ok = client.post(
        "/login",
        data={"username": "casey", "password": "newpass99"},
        follow_redirects=True,
    )
    assert "Alert console" in ok.get_data(as_text=True)


def test_forgot_password_unknown_identity_does_not_reveal(client):
    rv = client.post(
        "/forgot-password",
        data={"identifier": "nobody@example.com"},
        follow_redirects=True,
    )
    body = rv.get_data(as_text=True)
    assert "Email delivery is not configured" in body
    assert "email sent" not in body.lower()
    assert "reset-password?token=" not in body


def test_expired_reset_token_rejected(app, client):
    store: UserStore = app.extensions["user_store"]
    user = store.create_user(
        username="dana",
        email="dana@example.com",
        password="secret123",
    )
    token = store.create_reset_token(user.id, ttl_minutes=45)
    expired = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    with store._conn() as conn:
        conn.execute(
            "UPDATE users SET reset_expires = ? WHERE id = ?",
            (expired, user.id),
        )

    rv = client.get(f"/reset-password?token={token}", follow_redirects=True)
    assert "invalid or has expired" in rv.get_data(as_text=True)


def test_send_resend_sets_user_agent_and_reports_failure():
    class FakeResp:
        status = 200

        def read(self):
            return b'{"id":"re_test"}'

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    captured = {}

    def fake_urlopen(req, timeout=20):
        headers = {k.lower(): v for k, v in req.header_items()}
        captured["ua"] = headers.get("user-agent")
        captured["url"] = req.full_url
        return FakeResp()

    with patch("alerts.notify.urllib.request.urlopen", fake_urlopen):
        ok, detail = send_resend_email(
            api_key="re_test",
            from_addr="Mun Cyber Eye <ops@example.com>",
            to="alice@example.com",
            subject="Reset",
            html="<p>reset</p>",
        )
    assert ok is True
    assert detail == "re_test"
    assert captured["ua"] == DEFAULT_USER_AGENT

    missing_key = send_resend_email(
        api_key="",
        from_addr="ops@example.com",
        to="alice@example.com",
        subject="Reset",
        html="<p>x</p>",
    )
    assert missing_key == (False, "RESEND_API_KEY is not configured")


def test_developer_env_seed_when_both_set(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DEVELOPER_USERNAME": "muncyber",
            "DEVELOPER_PASSWORD": "lab-secret-99",
            "DEVELOPER_EMAIL": "lab@muncyber.example",
            "ADMIN_SYNC_PASSWORD": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        }
    )
    store: UserStore = application.extensions["user_store"]
    user = store.get_by_username("muncyber")
    assert user is not None
    assert user.role == "developer"
    client = application.test_client()
    ok = client.post(
        "/login",
        data={"username": "muncyber", "password": "lab-secret-99"},
        follow_redirects=True,
    )
    assert ok.status_code == 200
    text = ok.get_data(as_text=True)
    assert "Alert console" in text
    assert ">Train models</a>" in text
    page = client.get("/admin/train")
    assert page.status_code == 200
    assert "Model training" in page.get_data(as_text=True)


def test_developer_not_seeded_when_password_empty(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DEVELOPER_USERNAME": "muncyber",
            "DEVELOPER_PASSWORD": "",
            "ADMIN_SYNC_PASSWORD": False,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        }
    )
    store: UserStore = application.extensions["user_store"]
    assert store.get_by_username("muncyber") is None
    assert store.count() == 0


def test_admin_role_env_cannot_grant_developer(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "siteadmin",
            "ADMIN_PASSWORD": "test-pass-12",
            "ADMIN_EMAIL": "siteadmin@localhost",
            "ADMIN_ROLE": "developer",
            "ADMIN_SYNC_PASSWORD": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
        }
    )
    user = application.extensions["user_store"].get_by_username("siteadmin")
    assert user is not None
    assert user.role == "admin"
    client = application.test_client()
    client.post(
        "/login",
        data={"username": "siteadmin", "password": "test-pass-12"},
        follow_redirects=True,
    )
    dash = client.get("/")
    assert ">Train models</a>" not in dash.get_data(as_text=True)


def test_legacy_auth_db_accepts_developer_role(tmp_path):
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
        """
    )
    conn.commit()
    conn.close()
    store = UserStore(db)
    user = store.create_user(
        username="labdev",
        email="labdev@localhost",
        password="dev-pass-99",
        role="developer",
    )
    assert user.role == "developer"
    reloaded = store.get_by_username("labdev")
    assert reloaded is not None
    assert reloaded.role == "developer"

