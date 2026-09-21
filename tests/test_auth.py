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


def _register(client, username="alice", email="alice@example.com", password="test-pass-12"):
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
        data={"username": "alice", "password": "test-pass-12"},
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
            "password": "test-pass-12",
            "confirm_password": "test-pass-12",
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
            "password": "test-pass-12",
            "confirm_password": "test-pass-12",
        },
    )
    assert "already taken" in again.get_data(as_text=True)


def test_forgot_password_token_reset(client, app):
    _register(client, username="casey", email="casey@example.com", password="old-pass-12x")
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
            "password": "new-pass-12x",
            "confirm_password": "new-pass-12x",
        },
        follow_redirects=True,
    )
    assert "Password updated" in reset.get_data(as_text=True)

    still_old = client.post(
        "/login",
        data={"username": "casey", "password": "old-pass-12x"},
    )
    assert "Invalid credentials" in still_old.get_data(as_text=True)

    ok = client.post(
        "/login",
        data={"username": "casey", "password": "new-pass-12x"},
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
        password="test-pass-12",
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
        password="lab-secret-99",
        role="developer",
    )
    assert user.role == "developer"
    reloaded = store.get_by_username("labdev")
    assert reloaded is not None
    assert reloaded.role == "developer"


def test_register_page_states_password_rules(client):
    rv = client.get("/register")
    body = rv.get_data(as_text=True)
    assert "at least 12 characters" in body.lower()
    assert "letter" in body.lower()
    assert "digit" in body.lower()
    assert "changeme" in body.lower()
    assert "two-factor authentication is not available" in body.lower()
    assert 'minlength="12"' in body


def test_password_policy_rejects_weak_values():
    from app.auth import validate_password

    assert "at least 12" in validate_password("short1")
    assert "letter and one digit" in validate_password("abcdefghijkl")
    assert "letter and one digit" in validate_password("123456789012")
    for weak in (
        "changeme",
        "changeme1234",
        "password",
        "password123",
        "Password1234",
        "admin",
        "operator",
        "operator1234",
        "qwerty123456",
        "abcdef123456",
        "123456abcdef",
    ):
        err = validate_password(weak)
        assert err, f"expected reject for {weak!r}"
        assert "common" in err.lower() or "easy to guess" in err.lower() or "at least 12" in err
    assert validate_password("test-pass-12") is None
    assert validate_password("lab-secret-99") is None
    assert "do not match" in validate_password("test-pass-12", "other-pass-12").lower()


def test_register_rejects_changeme_and_short_passwords(client):
    short = client.post(
        "/register",
        data={
            "username": "newbie",
            "email": "newbie@example.com",
            "password": "changeme",
            "confirm_password": "changeme",
        },
    )
    assert short.status_code == 200
    body = short.get_data(as_text=True)
    assert "too common" in body.lower() or "at least 12" in body.lower()
    store_fail = client.post(
        "/register",
        data={
            "username": "newbie",
            "email": "newbie@example.com",
            "password": "password123",
            "confirm_password": "password123",
        },
    )
    assert "too common" in store_fail.get_data(as_text=True).lower() or "at least 12" in store_fail.get_data(as_text=True).lower()


def test_env_example_does_not_seed_operator_changeme():
    from pathlib import Path

    text = Path(__file__).resolve().parent.parent.joinpath(".env.example").read_text()
    assert "ADMIN_USERNAME=\n" in text
    assert "ADMIN_PASSWORD=\n" in text
    assert "DEVELOPER_USERNAME=\n" in text
    assert "DEVELOPER_PASSWORD=\n" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        assert not stripped.startswith("ADMIN_USERNAME=operator")
        assert not stripped.startswith("OPERATOR_USERNAME=operator")
        assert "PASSWORD=changeme" not in stripped.replace(" ", "")


def _insert_legacy_operator(db_path, *, password="changeme", email="operator@localhost", role="admin"):
    from werkzeug.security import generate_password_hash

    store = UserStore(db_path)
    with store._conn() as conn:
        conn.execute(
            """
            INSERT INTO users (
                id, username, email, password_hash, role, active,
                created_at, approved, approved_at, approved_by
            ) VALUES (?, 'operator', ?, ?, ?, 1, '2024-01-01T00:00:00Z',
                      1, '2024-01-01T00:00:00Z', 'legacy')
            """,
            ("legacy-op", email, generate_password_hash(password), role),
        )
    return store


def test_legacy_operator_changeme_login_always_rejected(tmp_path):
    db = tmp_path / "auth.db"
    store = _insert_legacy_operator(db)
    user, status = store.attempt_login("operator", "changeme")
    assert user is None
    assert status == "invalid"
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DEVELOPER_USERNAME": "",
            "DEVELOPER_PASSWORD": "",
            "DISABLE_LEGACY_OPERATOR": False,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(db),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    leftover = application.extensions["user_store"].get_by_username("operator")
    assert leftover is not None
    assert leftover.active is True
    client = application.test_client()
    rv = client.post("/login", data={"username": "operator", "password": "changeme"})
    assert rv.status_code == 200
    assert "Invalid credentials" in rv.get_data(as_text=True)
    assert "Alert console" not in rv.get_data(as_text=True)


def test_boot_deactivates_legacy_operator_changeme(tmp_path):
    db = tmp_path / "auth.db"
    _insert_legacy_operator(db, password="changeme", email="ops@example.com", role="admin")
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DISABLE_LEGACY_OPERATOR": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(db),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    store: UserStore = application.extensions["user_store"]
    user = store.get_by_username("operator")
    assert user is not None
    assert user.active is False
    actions = [row["action"] for row in application.extensions["alert_store"].list_system_audit()]
    assert "disable_legacy_operator" in actions
    client = application.test_client()
    rv = client.post("/login", data={"username": "operator", "password": "changeme"})
    assert "Invalid credentials" in rv.get_data(as_text=True)


def test_boot_deactivates_operator_localhost_email_even_with_strong_password(tmp_path):
    db = tmp_path / "auth.db"
    _insert_legacy_operator(
        db, password="test-pass-12", email="operator@localhost", role="admin"
    )
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DISABLE_LEGACY_OPERATOR": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(db),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    user = application.extensions["user_store"].get_by_username("operator")
    assert user is not None
    assert user.active is False


def test_intentional_operator_username_with_strong_password_stays_active(tmp_path):
    db = tmp_path / "auth.db"
    _insert_legacy_operator(
        db, password="test-pass-12", email="ops@example.com", role="operator"
    )
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "DISABLE_LEGACY_OPERATOR": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(db),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    store: UserStore = application.extensions["user_store"]
    user = store.get_by_username("operator")
    assert user is not None
    assert user.active is True
    client = application.test_client()
    denied = client.post("/login", data={"username": "operator", "password": "changeme"})
    assert "Invalid credentials" in denied.get_data(as_text=True)
    ok = client.post(
        "/login",
        data={"username": "operator", "password": "test-pass-12"},
        follow_redirects=True,
    )
    assert "Alert console" in ok.get_data(as_text=True)


def test_env_seed_refuses_operator_changeme(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "operator",
            "ADMIN_PASSWORD": "changeme",
            "ADMIN_EMAIL": "operator@localhost",
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    store: UserStore = application.extensions["user_store"]
    assert store.get_by_username("operator") is None
    assert store.count() == 0


def test_env_seed_operator_localhost_email_is_deactivated(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "OPERATOR_USERNAME": "operator",
            "OPERATOR_PASSWORD": "test-pass-12",
            "DISABLE_LEGACY_OPERATOR": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    user = application.extensions["user_store"].get_by_username("operator")
    assert user is not None
    assert user.email == "operator@localhost"
    assert user.active is False


def test_env_seed_operator_custom_email_stays_active(tmp_path):
    application = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret",
            "ADMIN_USERNAME": "",
            "ADMIN_PASSWORD": "",
            "OPERATOR_USERNAME": "operator",
            "OPERATOR_PASSWORD": "test-pass-12",
            "OPERATOR_EMAIL": "ops@example.com",
            "DISABLE_LEGACY_OPERATOR": True,
            "ALERT_DB_PATH": str(tmp_path / "alerts.db"),
            "AUTH_DB_PATH": str(tmp_path / "auth.db"),
            "SNAPSHOT_DIR": str(tmp_path / "snapshots"),
            "SEED_DEMO_CAMERAS": False,
        }
    )
    user = application.extensions["user_store"].get_by_username("operator")
    assert user is not None
    assert user.active is True


def test_create_user_and_reset_enforce_password_policy(tmp_path):
    store = UserStore(tmp_path / "auth.db")
    with pytest.raises(ValueError, match="at least 12|too common"):
        store.create_user(
            username="pat",
            email="pat@example.com",
            password="changeme",
        )
    user = store.create_user(
        username="pat",
        email="pat@example.com",
        password="old-pass-12x",
    )
    with pytest.raises(ValueError, match="at least 12|too common"):
        store.set_password(user.id, "password123")
    token = store.create_reset_token(user.id)
    with pytest.raises(ValueError, match="at least 12|too common"):
        store.consume_reset_token(token, "admin")
    token = store.create_reset_token(user.id)
    store.consume_reset_token(token, "new-pass-12x")
    ok, status = store.attempt_login("pat", "new-pass-12x")
    assert status == "ok"
    assert ok is not None

