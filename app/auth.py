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
APPROVER_ROLES = frozenset({"admin", DEVELOPER_ROLE})
USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
MIN_PASSWORD_LEN = 8
_USERS_ROLE_CHECK = "role IN ('admin', 'operator', 'developer')"
PENDING_LOGIN_MESSAGE = (
    "Your account is awaiting admin approval. You cannot sign in to the "
    "console, cameras, or alerts until a site admin or developer approves it."
)
INACTIVE_LOGIN_MESSAGE = (
    "This account is inactive. Contact a site admin if you need access restored."
)


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


def _row_bool(row: sqlite3.Row, key: str, default: bool = False) -> bool:
    if key not in row.keys() or row[key] is None:
        return default
    return bool(row[key])


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
    approved: bool = True
    approved_at: Optional[str] = None
    approved_by: str = ""

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
                    approved_by TEXT NOT NULL DEFAULT ''
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
                approved_by TEXT NOT NULL DEFAULT ''
            );
            INSERT INTO users_role_mig (
                id, username, email, password_hash, role, active,
                created_at, reset_token, reset_expires, security_email,
                approved, approved_at, approved_by
            )
            SELECT id, username, email, password_hash, role, active,
                   created_at, reset_token, reset_expires,
                   COALESCE(security_email, ''),
                   COALESCE(approved, 1),
                   approved_at,
                   COALESCE(approved_by, '')
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
                    reset_token = NULL, reset_expires = NULL
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
                    reset_token = NULL, reset_expires = NULL
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
                SET active = ?, reset_token = NULL, reset_expires = NULL
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
        )


def console_session_guard():
    """Return a redirect if the session is missing, pending, or inactive."""
    from flask import current_app

    username = session.get("user")
    if not username:
        return redirect(url_for("main.login", next=request.path))
    store = current_app.extensions.get("user_store")
    if store is None:
        return None
    user = store.get_by_username(username)
    if user is None or not user.can_access_console():
        session.clear()
        if user is not None and not user.approved:
            flash(PENDING_LOGIN_MESSAGE, "error")
        else:
            flash(INACTIVE_LOGIN_MESSAGE, "error")
        return redirect(url_for("main.login"))
    session["role"] = user.role
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
            flash(
                "Only a site admin or Mun Cyber developer can approve accounts.",
                "error",
            )
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
