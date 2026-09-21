"""Local operator accounts for the Mun Cyber Eye console.

SQLite is the source of truth after optional env bootstrap users are seeded.
Session keys stay ``user`` (username) and ``role``.

Customer ``admin`` / ``operator`` run detection and review. Only ``developer``
(Mun Cyber lab, env-seeded) may train models.
"""

from __future__ import annotations

import re
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import wraps
from pathlib import Path
from typing import Iterator, List, Optional

from flask import flash, redirect, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

# Customer site roles (Create account / ADMIN_* bootstrap). Never granted training.
CUSTOMER_ROLES = frozenset({"admin", "operator"})
DEVELOPER_ROLE = "developer"
VALID_ROLES = CUSTOMER_ROLES | {DEVELOPER_ROLE}
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LEN = 8
_USERS_ROLE_CHECK = "role IN ('admin', 'operator', 'developer')"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normalize_username(value: str) -> str:
    return value.strip()


def normalize_email(value: str) -> str:
    return value.strip().lower()


def normalize_optional_email(value: str) -> str:
    cleaned = normalize_email(value or "")
    return cleaned if cleaned and "@" in cleaned else ""


def account_notify_email(user: object) -> str:
    """Alert address: security_email if set, else login email. Never invent one."""
    security = normalize_optional_email(getattr(user, "security_email", "") or "")
    if security:
        return security
    login = normalize_optional_email(getattr(user, "email", "") or "")
    return login


def _row_text(row: sqlite3.Row, key: str, default: str = "") -> str:
    if key not in row.keys():
        return default
    value = row[key]
    return default if value is None else str(value)


def validate_username(username: str) -> Optional[str]:
    if not USERNAME_RE.match(username):
        return "Username must be 3–32 characters: letters, numbers, dot, underscore, or hyphen."
    return None


def validate_email(email: str) -> Optional[str]:
    if "@" not in email or " " in email:
        return "Enter a valid email address."
    if len(email) > 254:
        return "Email is too long."
    local, _, domain = email.partition("@")
    if not local or not domain:
        return "Enter a valid email address."
    if domain != "localhost" and "." not in domain:
        return "Enter a valid email address."
    return None


def validate_password(password: str, confirm: str | None = None) -> Optional[str]:
    if len(password) < MIN_PASSWORD_LEN:
        return f"Password must be at least {MIN_PASSWORD_LEN} characters."
    if confirm is not None and password != confirm:
        return "Passwords do not match."
    return None


@dataclass
class User:
    id: str
    username: str
    email: str
    password_hash: str
    role: str
    active: bool
    created_at: str
    reset_token: Optional[str] = None
    reset_expires: Optional[str] = None
    security_email: str = ""

    def notify_address(self) -> str:
        """Prefer security_email for alerts; otherwise the login email."""
        return account_notify_email(self)


class UserStore:
    """SQLite user directory for console login, registration, and resets."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL CHECK ({_USERS_ROLE_CHECK}),
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    reset_token TEXT,
                    reset_expires TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token);
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
            if "security_email" not in cols:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN security_email TEXT NOT NULL DEFAULT ''"
                )
            self._migrate_role_check(conn)

    @staticmethod
    def _migrate_role_check(conn: sqlite3.Connection) -> None:
        """Rebuild users when an older CHECK omitted developer."""
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        ).fetchone()
        sql = (row[0] or "") if row else ""
        if "developer" in sql.lower():
            return
        conn.executescript(
            f"""
            CREATE TABLE users_role_mig (
                id TEXT PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK ({_USERS_ROLE_CHECK}),
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                reset_token TEXT,
                reset_expires TEXT,
                security_email TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO users_role_mig (
                id, username, email, password_hash, role, active,
                created_at, reset_token, reset_expires, security_email
            )
            SELECT id, username, email, password_hash, role, active,
                   created_at, reset_token, reset_expires,
                   COALESCE(security_email, '')
            FROM users;
            DROP TABLE users;
            ALTER TABLE users_role_mig RENAME TO users;
            CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token);
            """
        )

    def ensure_env_user(
        self,
        username: str,
        password: str,
        email: str,
        *,
        role: str = "admin",
        sync_password: bool = True,
    ) -> User:
        username = normalize_username(username)
        email = normalize_email(email)
        role = (role or "operator").strip().lower()
        if not username:
            raise ValueError("Username is empty.")
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        existing = self.get_by_username(username)
        if existing is None:
            return self.create_user(
                username=username,
                email=email,
                password=password,
                role=role,
            )
        if sync_password:
            self.set_password(existing.id, password)
            with self._conn() as conn:
                conn.execute(
                    "UPDATE users SET email = ?, role = ? WHERE id = ?",
                    (email, role, existing.id),
                )
            # Do not overwrite security_email on env sync.
            refreshed = self.get_by_id(existing.id)
            assert refreshed is not None
            return refreshed
        return existing

    def ensure_bootstrap_admin(
        self,
        username: str,
        password: str,
        email: str,
        *,
        sync_password: bool = True,
        role: str = "admin",
    ) -> User:
        return self.ensure_env_user(
            username,
            password,
            email,
            role=role,
            sync_password=sync_password,
        )

    def create_user(
        self,
        *,
        username: str,
        email: str,
        password: str,
        role: str = "operator",
    ) -> User:
        username = normalize_username(username)
        email = normalize_email(email)
        role = role.strip().lower()
        if role not in VALID_ROLES:
            raise ValueError(f"Invalid role: {role}")
        err = validate_username(username) or validate_email(email)
        if err:
            raise ValueError(err)
        if not password:
            raise ValueError("Password is required.")
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
            active=True,
            created_at=_utc_stamp(),
        )
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        id, username, email, password_hash, role, active,
                        created_at, reset_token, reset_expires
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL)
                    """,
                    (
                        user.id,
                        user.username,
                        user.email,
                        user.password_hash,
                        user.role,
                        1,
                        user.created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            msg = str(exc).lower()
            if "username" in msg:
                raise ValueError("That username is already taken.") from exc
            if "email" in msg:
                raise ValueError("That email is already registered.") from exc
            raise ValueError("Username or email is already registered.") from exc
        return user

    def get_by_id(self, user_id: str) -> Optional[User]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE id = ?", (user_id,)
            ).fetchone()
        return self._row_to_user(row) if row else None

    def get_by_username(self, username: str) -> Optional[User]:
        username = normalize_username(username)
        if not username:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE username = ? COLLATE NOCASE",
                (username,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def list_users(self, active_only: bool = False) -> List[User]:
        q = "SELECT * FROM users"
        if active_only:
            q += " WHERE active = 1"
        q += " ORDER BY username COLLATE NOCASE ASC"
        with self._conn() as conn:
            rows = conn.execute(q).fetchall()
        return [self._row_to_user(r) for r in rows]

    def count(self) -> int:
        with self._conn() as conn:
            row = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        return int(row[0]) if row else 0

    def get_by_email(self, email: str) -> Optional[User]:
        email = normalize_email(email)
        if not email:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
                (email,),
            ).fetchone()
        return self._row_to_user(row) if row else None

    def find_by_username_or_email(self, identifier: str) -> Optional[User]:
        identifier = identifier.strip()
        if not identifier:
            return None
        if "@" in identifier:
            return self.get_by_email(identifier) or self.get_by_username(identifier)
        return self.get_by_username(identifier) or self.get_by_email(identifier)

    def authenticate(self, username: str, password: str) -> Optional[User]:
        user = self.get_by_username(username)
        if user is None or not user.active:
            return None
        if not check_password_hash(user.password_hash, password):
            return None
        return user

    def set_security_email(self, user_id: str, security_email: str) -> User:
        cleaned = normalize_optional_email(security_email)
        if cleaned:
            err = validate_email(cleaned)
            if err:
                raise ValueError(err)
        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if not exists:
                raise KeyError(f"User not found: {user_id}")
            conn.execute(
                "UPDATE users SET security_email = ? WHERE id = ?",
                (cleaned, user_id),
            )
        refreshed = self.get_by_id(user_id)
        assert refreshed is not None
        return refreshed

    def set_password(self, user_id: str, password: str) -> None:
        if not password:
            raise ValueError("Password is required.")
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET password_hash = ?, reset_token = NULL, reset_expires = NULL
                WHERE id = ?
                """,
                (generate_password_hash(password), user_id),
            )

    def create_reset_token(self, user_id: str, ttl_minutes: int = 45) -> str:
        token = secrets.token_urlsafe(32)
        expires = _utc_stamp(_utc_now() + timedelta(minutes=int(ttl_minutes)))
        with self._conn() as conn:
            conn.execute(
                "UPDATE users SET reset_token = ?, reset_expires = ? WHERE id = ?",
                (token, expires, user_id),
            )
        return token

    def get_by_reset_token(self, token: str) -> Optional[User]:
        token = (token or "").strip()
        if not token:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM users WHERE reset_token = ?", (token,)
            ).fetchone()
        if not row:
            return None
        user = self._row_to_user(row)
        if not user.active or not user.reset_expires:
            return None
        if _parse_utc(user.reset_expires) < _utc_now():
            return None
        return user

    def consume_reset_token(self, token: str, new_password: str) -> User:
        user = self.get_by_reset_token(token)
        if user is None:
            raise ValueError("Reset link is invalid or has expired.")
        err = validate_password(new_password)
        if err:
            raise ValueError(err)
        self.set_password(user.id, new_password)
        refreshed = self.get_by_id(user.id)
        assert refreshed is not None
        return refreshed

    @staticmethod
    def _row_to_user(row: sqlite3.Row) -> User:
        return User(
            id=row["id"],
            username=row["username"],
            email=row["email"],
            password_hash=row["password_hash"],
            role=row["role"],
            active=bool(row["active"]),
            created_at=row["created_at"],
            reset_token=row["reset_token"],
            reset_expires=row["reset_expires"],
            security_email=_row_text(row, "security_email"),
        )


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def can_train(role: str | None) -> bool:
    """True only for Mun Cyber developer accounts — never customer admin/operator."""
    return (role or "").strip().lower() == DEVELOPER_ROLE


def public_register_role(*, is_first: bool) -> str:
    """Create account roles: first user is site admin; later users are operators."""
    return "admin" if is_first else "operator"


def admin_required(view):
    """Site admin only — customer account directory, not model training."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        if session.get("role") != "admin":
            flash("Only the site admin role can manage accounts.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def developer_required(view):
    """Developer role only — train, activate, upload labeled frames, extract video."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        if not can_train(session.get("role")):
            flash(
                "Only Mun Cyber developer accounts can train or activate models. "
                "Site admin and operator accounts run detection and review; "
                "they cannot retrain checkpoints. Public Create account never "
                "grants the developer role.",
                "error",
            )
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def guest_only(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user"):
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def start_session(user: User) -> None:
    session["user"] = user.username
    session["role"] = user.role


def safe_next_url(default_endpoint: str = "main.dashboard") -> str:
    nxt = request.args.get("next") or request.form.get("next") or ""
    if nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return url_for(default_endpoint)
