"""Local operator accounts for the Mun Cyber Eye console.

SQLite is the source of truth after optional env bootstrap users are seeded.
Session keys stay ``user`` (username) and ``role``. Customer console sessions
also carry ``email_2fa_ok`` after a successful email sign-in code. Password-only
sessions (developer skip, or customer 2FA turned off) do not.

Customer ``admin`` / ``operator`` run detection and review. Only ``developer``
(Mun Cyber lab, env-seeded) may train models.
"""

from __future__ import annotations

import os
import re
import secrets
import sqlite3
import subprocess
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
APPROVER_ROLES = frozenset({"admin", DEVELOPER_ROLE})
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LEN = 8
_USERS_ROLE_CHECK = "role IN ('admin', 'operator', 'developer')"
PENDING_LOGIN_MESSAGE = "Account pending approval."
INACTIVE_LOGIN_MESSAGE = (
    "This account is inactive. Contact a site admin if you need access restored."
)
PENDING_2FA_SESSION_KEY = "pending_2fa_user"
PENDING_2FA_NEXT_KEY = "pending_2fa_next"
EMAIL_2FA_OK_KEY = "email_2fa_ok"
EMAIL_2FA_AT_KEY = "email_2fa_at"
SESSION_STARTED_KEY = "session_started_at"
SESSION_LAST_ACTIVITY_KEY = "session_last_activity_at"
SESSION_EPOCH_KEY = "session_epoch"
SESSION_HOURS_DEFAULT = 8
SESSION_IDLE_MINUTES_DEFAULT = 15
NEEDS_EMAIL_2FA_MESSAGE = (
    "Sign in again. An email sign-in code is required before the console opens."
)
SESSION_EXPIRED_MESSAGE = "Your sign-in session expired. Sign in again."
SESSION_REAUTH_MESSAGE = "Sign in again to continue."
LOGIN_CODE_DIGITS = 6
LOGIN_CODE_MINUTES_DEFAULT = 10
LOGIN_CODE_RESEND_SECONDS_DEFAULT = 45
LOGIN_CODE_MAX_ATTEMPTS = 5
# Sign-in codes go to users.email (Create account / env seed), not security_email.
LOGIN_CODE_EMAIL_NOTE = (
    "Sign-in codes are sent to the login email on this account."
)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_stamp(dt: Optional[datetime] = None) -> str:
    return (dt or _utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _read_text_line(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    for line in text.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned
    return ""


def _git_describe(root: Path) -> str:
    """Commit identity for source checkouts. Empty when git metadata is absent."""
    if not (root / ".git").exists():
        return ""
    try:
        proc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--abbrev=12"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    if proc.returncode != 0:
        return ""
    line = (proc.stdout or "").strip().splitlines()
    return line[0].strip() if line else ""


def resolve_app_session_epoch(root: str | Path, override: str | None = None) -> str:
    """Epoch stamped on new sign-ins and checked on later requests.

    A non-empty ``APP_SESSION_EPOCH`` override wins. Otherwise the value is
    the ``BUILD_EPOCH`` file written into Setup.exe and standalone zip builds.
    A source checkout with no baked file uses ``installer/VERSION`` plus
    ``git describe`` so a new commit invalidates older cookies.
    """
    if override is None:
        override = os.getenv("APP_SESSION_EPOCH", "")
    chosen = str(override or "").strip()
    if chosen:
        return chosen
    root_path = Path(root)
    baked = _read_text_line(root_path / "BUILD_EPOCH")
    if baked:
        return baked
    version = _read_text_line(root_path / "installer" / "VERSION")
    described = _git_describe(root_path)
    if version and described:
        return f"{version}+{described}"
    if described:
        return described
    if version:
        return version
    return "dev"


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


def _row_bool(row: sqlite3.Row, key: str, default: bool = False) -> bool:
    if key not in row.keys() or row[key] is None:
        return default
    return bool(row[key])


def _row_int(row: sqlite3.Row, key: str, default: int = 0) -> int:
    if key not in row.keys() or row[key] is None:
        return default
    try:
        return int(row[key])
    except (TypeError, ValueError):
        return default


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


class LoginCodeCooldown(ValueError):
    """Raised when a new sign-in code is requested too soon."""

    def __init__(self, seconds: int) -> None:
        self.seconds = max(1, int(seconds))
        super().__init__(
            f"Wait {self.seconds} seconds before requesting a new sign-in code."
        )


def normalize_login_code(value: str) -> str:
    return "".join(ch for ch in (value or "") if ch.isdigit())


def generate_login_code(digits: int = LOGIN_CODE_DIGITS) -> str:
    span = 10 ** int(digits)
    return f"{secrets.randbelow(span):0{int(digits)}d}"


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
    approved: bool = True
    approved_at: Optional[str] = None
    approved_by: str = ""
    login_code_hash: Optional[str] = None
    login_code_expires: Optional[str] = None
    login_code_created_at: Optional[str] = None
    login_code_attempts: int = 0

    def notify_address(self) -> str:
        """Prefer security_email for alerts; otherwise the login email."""
        return account_notify_email(self)

    def can_access_console(self) -> bool:
        """True when the account may sign in and use cameras / alerts / run."""
        return bool(self.active and self.approved)

    def access_status(self) -> str:
        if not self.approved and self.active:
            return "pending"
        if not self.approved:
            return "rejected"
        if not self.active:
            return "inactive"
        return "active"


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
                    reset_expires TEXT,
                    security_email TEXT NOT NULL DEFAULT '',
                    approved INTEGER NOT NULL DEFAULT 0,
                    approved_at TEXT,
                    approved_by TEXT NOT NULL DEFAULT '',
                    login_code_hash TEXT,
                    login_code_expires TEXT,
                    login_code_created_at TEXT,
                    login_code_attempts INTEGER NOT NULL DEFAULT 0
                );

                CREATE INDEX IF NOT EXISTS idx_users_reset_token ON users(reset_token);
                """
            )
            cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
            if "security_email" not in cols:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN security_email TEXT NOT NULL DEFAULT ''"
                )
            if "approved" not in cols:
                # Existing installs keep access; public register sets this explicitly.
                conn.execute(
                    "ALTER TABLE users ADD COLUMN approved INTEGER NOT NULL DEFAULT 1"
                )
            if "approved_at" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN approved_at TEXT")
            if "approved_by" not in cols:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN approved_by TEXT NOT NULL DEFAULT ''"
                )
            if "login_code_hash" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN login_code_hash TEXT")
            if "login_code_expires" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN login_code_expires TEXT")
            if "login_code_created_at" not in cols:
                conn.execute("ALTER TABLE users ADD COLUMN login_code_created_at TEXT")
            if "login_code_attempts" not in cols:
                conn.execute(
                    "ALTER TABLE users ADD COLUMN login_code_attempts "
                    "INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute(
                """
                UPDATE users
                SET approved_at = created_at,
                    approved_by = CASE
                        WHEN approved_by IS NULL OR approved_by = '' THEN 'legacy'
                        ELSE approved_by
                    END
                WHERE approved = 1
                  AND (approved_at IS NULL OR approved_at = '')
                """
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
                security_email TEXT NOT NULL DEFAULT '',
                approved INTEGER NOT NULL DEFAULT 0,
                approved_at TEXT,
                approved_by TEXT NOT NULL DEFAULT '',
                login_code_hash TEXT,
                login_code_expires TEXT,
                login_code_created_at TEXT,
                login_code_attempts INTEGER NOT NULL DEFAULT 0
            );
            INSERT INTO users_role_mig (
                id, username, email, password_hash, role, active,
                created_at, reset_token, reset_expires, security_email,
                approved, approved_at, approved_by,
                login_code_hash, login_code_expires, login_code_created_at,
                login_code_attempts
            )
            SELECT id, username, email, password_hash, role, active,
                   created_at, reset_token, reset_expires,
                   COALESCE(security_email, ''),
                   COALESCE(approved, 1),
                   approved_at,
                   COALESCE(approved_by, ''),
                   login_code_hash,
                   login_code_expires,
                   login_code_created_at,
                   COALESCE(login_code_attempts, 0)
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
                approved=True,
                approved_by="env",
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
        approved: bool = True,
        approved_by: str = "",
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
        created = _utc_stamp()
        is_approved = bool(approved)
        actor = (approved_by or "").strip()
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            email=email,
            password_hash=generate_password_hash(password),
            role=role,
            active=True,
            created_at=created,
            approved=is_approved,
            approved_at=created if is_approved else None,
            approved_by=actor if is_approved else "",
        )
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO users (
                        id, username, email, password_hash, role, active,
                        created_at, reset_token, reset_expires,
                        approved, approved_at, approved_by
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?)
                    """,
                    (
                        user.id,
                        user.username,
                        user.email,
                        user.password_hash,
                        user.role,
                        1,
                        user.created_at,
                        1 if user.approved else 0,
                        user.approved_at,
                        user.approved_by,
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

    def list_pending_users(self) -> List[User]:
        with self._conn() as conn:
            rows = conn.execute(
                """
                SELECT * FROM users
                WHERE approved = 0 AND active = 1
                ORDER BY created_at ASC, username COLLATE NOCASE ASC
                """
            ).fetchall()
        return [self._row_to_user(r) for r in rows]

    def count_approved_approvers(self, *, exclude_id: str | None = None) -> int:
        q = (
            "SELECT COUNT(*) FROM users WHERE approved = 1 AND active = 1 "
            "AND role IN ('admin', 'developer')"
        )
        params: list[object] = []
        if exclude_id:
            q += " AND id != ?"
            params.append(exclude_id)
        with self._conn() as conn:
            row = conn.execute(q, params).fetchone()
        return int(row[0]) if row else 0

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
        user, status = self.attempt_login(username, password)
        return user if status == "ok" else None

    def attempt_login(self, username: str, password: str) -> tuple[Optional[User], str]:
        """Return (user, status) where status is ok, invalid, pending, or inactive."""
        user = self.get_by_username(username)
        if user is None or not check_password_hash(user.password_hash, password):
            return None, "invalid"
        if not user.approved:
            if not user.active:
                return user, "inactive"
            return user, "pending"
        if not user.active:
            return user, "inactive"
        return user, "ok"

    def _require_user(self, user_id: str) -> User:
        user = self.get_by_id(user_id)
        if user is None:
            raise KeyError(f"User not found: {user_id}")
        return user

    def _ensure_not_last_approver(self, user: User) -> None:
        if user.role not in APPROVER_ROLES or not user.can_access_console():
            return
        if self.count_approved_approvers(exclude_id=user.id) == 0:
            raise ValueError(
                "Cannot reject or deactivate the last approved admin or developer."
            )

    def approve_user(self, user_id: str, actor: str) -> User:
        user = self._require_user(user_id)
        stamp = _utc_stamp()
        actor = (actor or "unknown").strip() or "unknown"
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET approved = 1, active = 1, approved_at = ?, approved_by = ?,
                    reset_token = NULL, reset_expires = NULL,
                    login_code_hash = NULL, login_code_expires = NULL,
                    login_code_created_at = NULL, login_code_attempts = 0
                WHERE id = ?
                """,
                (stamp, actor, user.id),
            )
        refreshed = self.get_by_id(user.id)
        assert refreshed is not None
        return refreshed

    def reject_user(self, user_id: str, actor: str) -> User:
        user = self._require_user(user_id)
        self._ensure_not_last_approver(user)
        actor = (actor or "unknown").strip() or "unknown"
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET approved = 0, active = 0, approved_by = ?,
                    reset_token = NULL, reset_expires = NULL,
                    login_code_hash = NULL, login_code_expires = NULL,
                    login_code_created_at = NULL, login_code_attempts = 0
                WHERE id = ?
                """,
                (actor, user.id),
            )
        refreshed = self.get_by_id(user.id)
        assert refreshed is not None
        return refreshed

    def set_user_active(self, user_id: str, active: bool) -> User:
        user = self._require_user(user_id)
        if not active:
            self._ensure_not_last_approver(user)
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET active = ?, reset_token = NULL, reset_expires = NULL,
                    login_code_hash = NULL, login_code_expires = NULL,
                    login_code_created_at = NULL, login_code_attempts = 0
                WHERE id = ?
                """,
                (1 if active else 0, user.id),
            )
        refreshed = self.get_by_id(user.id)
        assert refreshed is not None
        return refreshed

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
                SET password_hash = ?, reset_token = NULL, reset_expires = NULL,
                    login_code_hash = NULL, login_code_expires = NULL,
                    login_code_created_at = NULL, login_code_attempts = 0
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
        if not user.active or not user.approved or not user.reset_expires:
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

    def login_code_resend_wait_seconds(
        self, user: User, min_seconds: int
    ) -> int:
        if not user.login_code_hash or not user.login_code_created_at:
            return 0
        if user.login_code_expires:
            try:
                if _parse_utc(user.login_code_expires) < _utc_now():
                    return 0
            except ValueError:
                return 0
        try:
            created = _parse_utc(user.login_code_created_at)
        except ValueError:
            return 0
        remain = int(min_seconds - (_utc_now() - created).total_seconds())
        return remain if remain > 0 else 0

    def create_login_code(
        self,
        user_id: str,
        *,
        ttl_minutes: int = LOGIN_CODE_MINUTES_DEFAULT,
        min_resend_seconds: int = LOGIN_CODE_RESEND_SECONDS_DEFAULT,
        force: bool = False,
    ) -> str:
        """Issue a single-use email sign-in code. Never for pending accounts."""
        user = self._require_user(user_id)
        if not user.can_access_console():
            raise ValueError(
                "Sign-in codes are only issued after the account is approved."
            )
        if not (user.email or "").strip():
            raise ValueError("This account has no login email for a sign-in code.")
        if not force:
            wait = self.login_code_resend_wait_seconds(user, int(min_resend_seconds))
            if wait > 0:
                raise LoginCodeCooldown(wait)
        code = generate_login_code()
        expires = _utc_stamp(_utc_now() + timedelta(minutes=int(ttl_minutes)))
        created = _utc_stamp()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET login_code_hash = ?, login_code_expires = ?,
                    login_code_created_at = ?, login_code_attempts = 0
                WHERE id = ?
                """,
                (generate_password_hash(code), expires, created, user.id),
            )
        return code

    def clear_login_code(self, user_id: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE users
                SET login_code_hash = NULL, login_code_expires = NULL,
                    login_code_created_at = NULL, login_code_attempts = 0
                WHERE id = ?
                """,
                (user_id,),
            )

    def verify_login_code(self, user_id: str, code: str) -> User:
        user = self._require_user(user_id)
        if not user.can_access_console():
            self.clear_login_code(user.id)
            raise ValueError(
                "Sign-in codes are only accepted after the account is approved."
            )
        code = normalize_login_code(code)
        if (
            not user.login_code_hash
            or not user.login_code_expires
            or len(code) != LOGIN_CODE_DIGITS
        ):
            raise ValueError("Sign-in code is invalid or has expired.")
        try:
            expired = _parse_utc(user.login_code_expires) < _utc_now()
        except ValueError:
            expired = True
        if expired:
            self.clear_login_code(user.id)
            raise ValueError("Sign-in code is invalid or has expired.")
        if user.login_code_attempts >= LOGIN_CODE_MAX_ATTEMPTS:
            self.clear_login_code(user.id)
            raise ValueError(
                "Too many incorrect codes. Sign in again to request a new one."
            )
        if not check_password_hash(user.login_code_hash, code):
            attempts = user.login_code_attempts + 1
            if attempts >= LOGIN_CODE_MAX_ATTEMPTS:
                self.clear_login_code(user.id)
                raise ValueError(
                    "Too many incorrect codes. Sign in again to request a new one."
                )
            with self._conn() as conn:
                conn.execute(
                    "UPDATE users SET login_code_attempts = ? WHERE id = ?",
                    (attempts, user.id),
                )
            raise ValueError("That sign-in code is incorrect.")
        self.clear_login_code(user.id)
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
            approved=_row_bool(row, "approved", default=True),
            approved_at=_row_text(row, "approved_at") or None,
            approved_by=_row_text(row, "approved_by"),
            login_code_hash=_row_text(row, "login_code_hash") or None,
            login_code_expires=_row_text(row, "login_code_expires") or None,
            login_code_created_at=_row_text(row, "login_code_created_at") or None,
            login_code_attempts=_row_int(row, "login_code_attempts", 0),
        )


def two_factor_required_for(user: Optional[User]) -> bool:
    """Email sign-in code after approval. Pending never. Developer off by default."""
    from flask import current_app

    if user is None or not user.can_access_console():
        return False
    role = (user.role or "").strip().lower()
    if role in CUSTOMER_ROLES:
        return bool(current_app.config.get("CUSTOMER_2FA_REQUIRED", True))
    if role == DEVELOPER_ROLE:
        return bool(current_app.config.get("DEVELOPER_2FA_REQUIRED", False))
    return False


def pending_login_code_username() -> str:
    return (session.get(PENDING_2FA_SESSION_KEY) or "").strip()


def _email_2fa_stamp_ok() -> bool:
    """True only after a successful email-code verify in this session."""
    if session.get(EMAIL_2FA_OK_KEY) is not True:
        return False
    stamp = session.get(EMAIL_2FA_AT_KEY)
    if not stamp:
        return False
    try:
        _parse_utc(str(stamp))
    except ValueError:
        return False
    return True


def _configured_session_hours() -> float:
    from flask import current_app

    raw = current_app.config.get("SESSION_HOURS", SESSION_HOURS_DEFAULT)
    try:
        hours = float(raw)
    except (TypeError, ValueError):
        hours = float(SESSION_HOURS_DEFAULT)
    if hours <= 0:
        return float(SESSION_HOURS_DEFAULT)
    return hours


def _session_lifetime_exceeded() -> bool:
    """Reject cookies with no start time or older than SESSION_HOURS."""
    stamp = session.get(SESSION_STARTED_KEY)
    if not stamp:
        return True
    try:
        started = _parse_utc(str(stamp))
    except ValueError:
        return True
    return _utc_now() - started > timedelta(hours=_configured_session_hours())


def _configured_idle_minutes() -> float:
    from flask import current_app

    raw = current_app.config.get("SESSION_IDLE_MINUTES", SESSION_IDLE_MINUTES_DEFAULT)
    try:
        minutes = float(raw)
    except (TypeError, ValueError):
        minutes = float(SESSION_IDLE_MINUTES_DEFAULT)
    if minutes <= 0:
        return float(SESSION_IDLE_MINUTES_DEFAULT)
    return minutes


def _session_idle_exceeded() -> bool:
    """True when the last authenticated request is older than the idle window."""
    stamp = session.get(SESSION_LAST_ACTIVITY_KEY)
    if not stamp:
        return True
    try:
        seen = _parse_utc(str(stamp))
    except ValueError:
        return True
    return _utc_now() - seen > timedelta(minutes=_configured_idle_minutes())


def _current_session_epoch() -> str:
    from flask import current_app

    return str(current_app.config.get("APP_SESSION_EPOCH") or "").strip()


def _session_epoch_mismatch() -> bool:
    """True when this cookie was issued for a different build than the one running."""
    current = _current_session_epoch()
    stamped = str(session.get(SESSION_EPOCH_KEY) or "").strip()
    if not current or not stamped:
        return True
    return stamped != current


def _request_is_static_asset() -> bool:
    """CSS, scripts, and images must not count as console activity."""
    endpoint = request.endpoint or ""
    if endpoint == "static" or endpoint.endswith(".static"):
        return True
    return (request.path or "").startswith("/static/")


def _reject_console_session(reason: str):
    """Drop a cookie that must not open the console, then send the user to login."""
    pending = reason == "pending"
    session.clear()
    if pending:
        flash(PENDING_LOGIN_MESSAGE, "error")
    elif reason == "needs_2fa":
        flash(NEEDS_EMAIL_2FA_MESSAGE, "error")
    elif reason == "expired":
        flash(SESSION_EXPIRED_MESSAGE, "error")
    elif reason in {"idle", "epoch"}:
        flash(SESSION_REAUTH_MESSAGE, "error")
    else:
        flash(INACTIVE_LOGIN_MESSAGE, "error")
    return redirect(url_for("main.login"))


def begin_login_code_challenge(user: User, next_url: str = "") -> None:
    session.pop("user", None)
    session.pop("role", None)
    session.pop(EMAIL_2FA_OK_KEY, None)
    session.pop(EMAIL_2FA_AT_KEY, None)
    session.pop(SESSION_STARTED_KEY, None)
    session.pop(SESSION_LAST_ACTIVITY_KEY, None)
    session.pop(SESSION_EPOCH_KEY, None)
    session[PENDING_2FA_SESSION_KEY] = user.username
    if next_url:
        session[PENDING_2FA_NEXT_KEY] = next_url
    else:
        session.pop(PENDING_2FA_NEXT_KEY, None)


def clear_login_code_challenge() -> None:
    session.pop(PENDING_2FA_SESSION_KEY, None)
    session.pop(PENDING_2FA_NEXT_KEY, None)
    session.pop("_demo_login_code", None)


def peek_demo_login_code() -> str:
    value = session.get("_demo_login_code")
    return str(value) if value else ""


def stash_demo_login_code(code: str) -> None:
    if code:
        session["_demo_login_code"] = code


def console_session_guard():
    """Return a redirect if the session is missing, pending, unverified, or stale.

    Customer admin and operator cookies must carry ``email_2fa_ok`` when
    ``two_factor_required_for`` is true. A password-only cookie from before
    email codes were required cannot open the console.

    Idle timeout is checked before the absolute ``SESSION_HOURS`` cap. A quiet
    console signs out at the idle limit; a console that stays in use still
    ends when the absolute age is reached. A cookie stamped for another build
    epoch is cleared even when it is still inside both windows. Static assets
    do not refresh the idle clock.
    """
    from flask import current_app

    username = session.get("user")
    if not username:
        if pending_login_code_username():
            return redirect(url_for("main.login_code"))
        return redirect(url_for("main.login", next=request.path))
    store = current_app.extensions.get("user_store")
    if store is None:
        return None
    user = store.get_by_username(username)
    if user is None or not user.can_access_console():
        pending = user is not None and not user.approved
        return _reject_console_session("pending" if pending else "inactive")
    if two_factor_required_for(user) and not _email_2fa_stamp_ok():
        return _reject_console_session("needs_2fa")
    if _session_epoch_mismatch():
        return _reject_console_session("epoch")
    if _session_idle_exceeded():
        return _reject_console_session("idle")
    if _session_lifetime_exceeded():
        return _reject_console_session("expired")
    session["role"] = user.role
    if not _request_is_static_asset():
        session[SESSION_LAST_ACTIVITY_KEY] = _utc_stamp()
    return None


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        blocked = console_session_guard()
        if blocked is not None:
            return blocked
        return view(*args, **kwargs)

    return wrapped


def can_train(role: str | None) -> bool:
    """True only for Mun Cyber developer accounts — never customer admin/operator."""
    return (role or "").strip().lower() == DEVELOPER_ROLE


def can_approve_accounts(role: str | None) -> bool:
    """True for site admin and Mun Cyber developer — not ordinary operators."""
    return (role or "").strip().lower() in APPROVER_ROLES


def public_register_role(*, is_first: bool) -> str:
    """Create account roles: first user is site admin; later users are operators."""
    return "admin" if is_first else "operator"


def admin_required(view):
    """Site admin only — kept for callers that must exclude developers."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        blocked = console_session_guard()
        if blocked is not None:
            return blocked
        if session.get("role") != "admin":
            flash("Only the site admin role can manage this page.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def approver_required(view):
    """Site admin or developer — approve / reject customer registrations."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        blocked = console_session_guard()
        if blocked is not None:
            return blocked
        if not can_approve_accounts(session.get("role")):
            flash("Only an administrator can approve accounts.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def developer_required(view):
    """Developer role only — train, activate, upload labeled frames, extract video."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        blocked = console_session_guard()
        if blocked is not None:
            return blocked
        if not can_train(session.get("role")):
            flash("Only developer accounts can train models.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def guest_only(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if session.get("user"):
            blocked = console_session_guard()
            if blocked is not None:
                return blocked
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def start_session(user: User, *, email_2fa_verified: bool = False) -> None:
    """Open a console session.

    ``email_2fa_verified`` is set only after a successful email-code check.
    Password-only sign-in (developer skip, or customer 2FA disabled) must
    leave ``email_2fa_ok`` unset so a later required-2FA policy rejects it.

    The cookie is a browser session cookie (not permanent). ``SESSION_HOURS``
    caps total age. ``SESSION_IDLE_MINUTES`` caps quiet time. ``session_epoch``
    must match the running build or the next request asks for sign-in again.
    """
    clear_login_code_challenge()
    now = _utc_stamp()
    session["user"] = user.username
    session["role"] = user.role
    session[SESSION_STARTED_KEY] = now
    session[SESSION_LAST_ACTIVITY_KEY] = now
    session[SESSION_EPOCH_KEY] = _current_session_epoch()
    session.permanent = False
    if email_2fa_verified:
        session[EMAIL_2FA_OK_KEY] = True
        session[EMAIL_2FA_AT_KEY] = now
    else:
        session.pop(EMAIL_2FA_OK_KEY, None)
        session.pop(EMAIL_2FA_AT_KEY, None)


def safe_next_url(default_endpoint: str = "main.dashboard") -> str:
    nxt = request.args.get("next") or request.form.get("next") or ""
    if nxt.startswith("/") and not nxt.startswith("//"):
        return nxt
    return url_for(default_endpoint)
