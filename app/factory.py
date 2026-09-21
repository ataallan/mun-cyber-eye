"""Flask application factory for Mun Cyber Eye console."""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, session

from alerts.notify import NotificationService, NotifyConfig
from alerts.store import AlertStore
from ingest.cameras import CameraStore

from .auth import CUSTOMER_ROLES, UserStore, can_approve_accounts, can_train


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _session_hours(value) -> float:
    """Console session cap. Non-positive or junk values fall back to 8 hours."""
    try:
        hours = float(value)
    except (TypeError, ValueError):
        hours = 8.0
    if hours <= 0:
        return 8.0
    return hours


def _customer_bootstrap_role(value: str | None) -> str:
    """ADMIN_* may seed site admin or operator — never developer / training."""
    role = (value or "admin").strip().lower()
    return role if role in CUSTOMER_ROLES else "admin"


def _seed_env_users(user_store: UserStore, config: dict) -> None:
    """Optional env bootstrap only when both username and password are set.

    Fresh installs ship with empty ADMIN_* and DEVELOPER_* values. There is no
    default password. The first Create account becomes customer site admin and
    cannot train. Developer (Mun Cyber lab) is seeded only via DEVELOPER_*.
    """
    sync = bool(config.get("ADMIN_SYNC_PASSWORD", True))
    admin_username = (config.get("ADMIN_USERNAME") or "").strip()
    admin_password = config.get("ADMIN_PASSWORD") or ""
    if admin_username and admin_password:
        user_store.ensure_bootstrap_admin(
            admin_username,
            admin_password,
            config.get("ADMIN_EMAIL") or f"{admin_username}@localhost",
            sync_password=sync,
            role=_customer_bootstrap_role(config.get("ADMIN_ROLE")),
        )
    op_user = (config.get("OPERATOR_USERNAME") or "").strip()
    op_pass = config.get("OPERATOR_PASSWORD") or ""
    if op_user and op_pass:
        user_store.ensure_env_user(
            op_user,
            op_pass,
            config.get("OPERATOR_EMAIL") or f"{op_user}@localhost",
            role="operator",
            sync_password=sync,
        )
    dev_user = (config.get("DEVELOPER_USERNAME") or "").strip()
    dev_pass = config.get("DEVELOPER_PASSWORD") or ""
    if dev_user and dev_pass:
        user_store.ensure_env_user(
            dev_user,
            dev_pass,
            config.get("DEVELOPER_EMAIL") or f"{dev_user}@localhost",
            role="developer",
            sync_password=sync,
        )


def create_app(test_config: dict | None = None) -> Flask:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    admin_username = (os.getenv("ADMIN_USERNAME") or "").strip()
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-only-change-me"),
        ADMIN_USERNAME=admin_username,
        ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD", ""),
        ADMIN_ROLE=os.getenv("ADMIN_ROLE", "admin"),
        ADMIN_EMAIL=os.getenv("ADMIN_EMAIL", "")
        or (f"{admin_username}@localhost" if admin_username else ""),
        ADMIN_SYNC_PASSWORD=_env_flag("ADMIN_SYNC_PASSWORD", "1"),
        OPERATOR_USERNAME=os.getenv("OPERATOR_USERNAME", ""),
        OPERATOR_PASSWORD=os.getenv("OPERATOR_PASSWORD", ""),
        OPERATOR_EMAIL=os.getenv("OPERATOR_EMAIL", ""),
        DEVELOPER_USERNAME=(os.getenv("DEVELOPER_USERNAME") or "").strip(),
        DEVELOPER_PASSWORD=os.getenv("DEVELOPER_PASSWORD", ""),
        DEVELOPER_EMAIL=os.getenv("DEVELOPER_EMAIL", ""),
        ALERT_DB_PATH=os.getenv("ALERT_DB_PATH", str(root / "data" / "alerts.db")),
        AUTH_DB_PATH=os.getenv("AUTH_DB_PATH", str(root / "data" / "auth.db")),
        CAMERA_DB_PATH=os.getenv("CAMERA_DB_PATH", str(root / "data" / "cameras.db")),
        SNAPSHOT_DIR=os.getenv("SNAPSHOT_DIR", str(root / "data" / "snapshots")),
        UPLOAD_DIR=os.getenv("UPLOAD_DIR", str(root / "data" / "uploads")),
        ALLOW_WEBCAM=_env_flag("ALLOW_WEBCAM", "0"),
        RTSP_CONNECT_TIMEOUT_SEC=float(os.getenv("RTSP_CONNECT_TIMEOUT_SEC", "8")),
        VISION_BACKEND=os.getenv("VISION_BACKEND", "auto"),
        ACTIVITY_CHECKPOINT=os.getenv(
            "ACTIVITY_CHECKPOINT",
            str(root / "data" / "checkpoints" / "activity_demo.joblib"),
        ),
        ACTIVITY_DATA_ROOT=os.getenv(
            "ACTIVITY_DATA_ROOT", str(root / "data" / "activity")
        ),
        OBJECTS_DATA_ROOT=os.getenv(
            "OBJECTS_DATA_ROOT", str(root / "data" / "objects")
        ),
        DANGEROUS_DATA_ROOT=os.getenv(
            "DANGEROUS_DATA_ROOT", str(root / "data" / "dangerous")
        ),
        CHECKPOINTS_DIR=os.getenv(
            "CHECKPOINTS_DIR", str(root / "data" / "checkpoints")
        ),
        ACTIVE_CHECKPOINT_FILE=os.getenv(
            "ACTIVE_CHECKPOINT_FILE",
            str(root / "data" / "active_checkpoint.json"),
        ),
        SAMPLE_FPS=float(os.getenv("SAMPLE_FPS", "2")),
        MAX_FRAMES_PER_RUN=int(os.getenv("MAX_FRAMES_PER_RUN", "120")),
        DEFAULT_CAMERA_LABEL=os.getenv(
            "DEFAULT_CAMERA_LABEL", "Authorized Camera — Demo Lab"
        ),
        DEFAULT_LOCATION_LABEL=os.getenv("DEFAULT_LOCATION_LABEL", "Demo Lab"),
        DEFAULT_CAMERA_ID=os.getenv("DEFAULT_CAMERA_ID", "demo-cam-01"),
        RESEND_API_KEY=os.getenv("RESEND_API_KEY", ""),
        RESEND_FROM=os.getenv(
            "RESEND_FROM", "Mun Cyber Technologies <info@muncyber.com>"
        ),
        ALERT_EMAIL_RECIPIENTS=os.getenv("ALERT_EMAIL_RECIPIENTS", ""),
        SECURITY_ALERT_EMAIL=os.getenv("SECURITY_ALERT_EMAIL", ""),
        ALERT_WEBHOOK_URL=os.getenv("ALERT_WEBHOOK_URL", ""),
        ALERT_WEBHOOK_SECRET=os.getenv("ALERT_WEBHOOK_SECRET", ""),
        ALERT_NOTIFY_ON_CREATE=os.getenv("ALERT_NOTIFY_ON_CREATE", "1") != "0",
        ALERT_NOTIFY_MAX_ATTEMPTS=int(os.getenv("ALERT_NOTIFY_MAX_ATTEMPTS", "3")),
        PROJECT_ROOT=str(root),
        PUBLIC_BASE_URL=os.getenv("PUBLIC_BASE_URL", "").rstrip("/"),
        RESET_TOKEN_MINUTES=int(os.getenv("RESET_TOKEN_MINUTES", "45")),
        AUTH_SHOW_RESET_URL=_env_flag("AUTH_SHOW_RESET_URL", "0"),
        AUTH_SHOW_LOGIN_CODE=_env_flag("AUTH_SHOW_LOGIN_CODE", "0"),
        CUSTOMER_2FA_REQUIRED=_env_flag("CUSTOMER_2FA_REQUIRED", "1"),
        DEVELOPER_2FA_REQUIRED=_env_flag("DEVELOPER_2FA_REQUIRED", "0"),
        LOGIN_CODE_MINUTES=int(os.getenv("LOGIN_CODE_MINUTES", "10")),
        LOGIN_CODE_RESEND_SECONDS=int(os.getenv("LOGIN_CODE_RESEND_SECONDS", "45")),
        SESSION_HOURS=_session_hours(os.getenv("SESSION_HOURS", "8")),
        ENABLE_FACE_AGGRESSION=_env_flag("ENABLE_FACE_AGGRESSION", "0"),
        ENABLE_GUNSHOT_AUDIO=_env_flag("ENABLE_GUNSHOT_AUDIO", "0"),
        ALERT_ON_INTENSE_SPORT=_env_flag("ALERT_ON_INTENSE_SPORT", "0"),
    )
    test_overrides = set(test_config or {})
    if test_config:
        app.config.update(test_config)
        if "CAMERA_DB_PATH" not in test_config and test_config.get("ALERT_DB_PATH"):
            app.config["CAMERA_DB_PATH"] = str(
                Path(app.config["ALERT_DB_PATH"]).with_name("cameras.db")
            )
        if "UPLOAD_DIR" not in test_config and test_config.get("SNAPSHOT_DIR"):
            app.config["UPLOAD_DIR"] = str(
                Path(app.config["SNAPSHOT_DIR"]).parent / "uploads"
            )
        if "OBJECTS_DATA_ROOT" not in test_config and test_config.get("ACTIVITY_DATA_ROOT"):
            app.config["OBJECTS_DATA_ROOT"] = str(
                Path(app.config["ACTIVITY_DATA_ROOT"]).parent / "objects"
            )
        if "DANGEROUS_DATA_ROOT" not in test_config and test_config.get("ACTIVITY_DATA_ROOT"):
            app.config["DANGEROUS_DATA_ROOT"] = str(
                Path(app.config["ACTIVITY_DATA_ROOT"]).parent / "dangerous"
            )
        # Existing login tests expect password → console. Opt in per test for email 2FA.
        if "CUSTOMER_2FA_REQUIRED" not in test_overrides:
            app.config["CUSTOMER_2FA_REQUIRED"] = False
        if "DEVELOPER_2FA_REQUIRED" not in test_overrides:
            app.config["DEVELOPER_2FA_REQUIRED"] = False

    def _abs(path_value: str) -> str:
        path = Path(path_value)
        return str(path if path.is_absolute() else root / path)

    app.config["ALERT_DB_PATH"] = _abs(app.config["ALERT_DB_PATH"])
    app.config["AUTH_DB_PATH"] = _abs(app.config["AUTH_DB_PATH"])
    app.config["CAMERA_DB_PATH"] = _abs(
        app.config.get("CAMERA_DB_PATH") or str(root / "data" / "cameras.db")
    )
    app.config["SNAPSHOT_DIR"] = _abs(app.config["SNAPSHOT_DIR"])
    app.config["UPLOAD_DIR"] = _abs(
        app.config.get("UPLOAD_DIR") or str(Path(app.config["SNAPSHOT_DIR"]).parent / "uploads")
    )
    app.config["ACTIVITY_DATA_ROOT"] = _abs(
        app.config.get("ACTIVITY_DATA_ROOT") or str(root / "data" / "activity")
    )
    app.config["OBJECTS_DATA_ROOT"] = _abs(
        app.config.get("OBJECTS_DATA_ROOT") or str(root / "data" / "objects")
    )
    app.config["DANGEROUS_DATA_ROOT"] = _abs(
        app.config.get("DANGEROUS_DATA_ROOT") or str(root / "data" / "dangerous")
    )
    app.config["CHECKPOINTS_DIR"] = _abs(
        app.config.get("CHECKPOINTS_DIR") or str(root / "data" / "checkpoints")
    )
    app.config["ACTIVE_CHECKPOINT_FILE"] = _abs(
        app.config.get("ACTIVE_CHECKPOINT_FILE")
        or str(root / "data" / "active_checkpoint.json")
    )
    if "ACTIVITY_CHECKPOINT" not in test_overrides:
        from vision.checkpoint_config import read_active_checkpoint

        record = read_active_checkpoint(app.config["ACTIVE_CHECKPOINT_FILE"])
        if record and record.get("path"):
            app.config["ACTIVITY_CHECKPOINT"] = record["path"]
    app.config["ACTIVITY_CHECKPOINT"] = _abs(app.config["ACTIVITY_CHECKPOINT"])
    app.config["ALLOW_WEBCAM"] = bool(app.config.get("ALLOW_WEBCAM", False))
    try:
        app.config["RTSP_CONNECT_TIMEOUT_SEC"] = float(
            app.config.get("RTSP_CONNECT_TIMEOUT_SEC", 8)
        )
    except (TypeError, ValueError):
        app.config["RTSP_CONNECT_TIMEOUT_SEC"] = 8.0
    try:
        minutes = app.config.get("LOGIN_CODE_MINUTES", 10)
        if minutes is None or minutes == "":
            minutes = 10
        app.config["LOGIN_CODE_MINUTES"] = int(minutes)
    except (TypeError, ValueError):
        app.config["LOGIN_CODE_MINUTES"] = 10
    try:
        resend_s = app.config.get("LOGIN_CODE_RESEND_SECONDS", 45)
        if resend_s is None or resend_s == "":
            resend_s = 45
        app.config["LOGIN_CODE_RESEND_SECONDS"] = int(resend_s)
    except (TypeError, ValueError):
        app.config["LOGIN_CODE_RESEND_SECONDS"] = 45

    session_hours = _session_hours(app.config.get("SESSION_HOURS", 8))
    app.config["SESSION_HOURS"] = session_hours
    # Applies only if a session is marked permanent. Sign-in keeps cookies
    # non-permanent; console_session_guard also enforces this cap.
    app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=session_hours)

    Path(app.config["ALERT_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["AUTH_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["CAMERA_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["SNAPSHOT_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["UPLOAD_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["ACTIVITY_DATA_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["OBJECTS_DATA_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["CHECKPOINTS_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(app.config["ACTIVE_CHECKPOINT_FILE"]).parent.mkdir(parents=True, exist_ok=True)

    store = AlertStore(app.config["ALERT_DB_PATH"])
    notify_config = NotifyConfig(
        resend_api_key=app.config.get("RESEND_API_KEY") or "",
        resend_from=app.config.get("RESEND_FROM")
        or "Mun Cyber Technologies <info@muncyber.com>",
        email_recipients_env=app.config.get("ALERT_EMAIL_RECIPIENTS") or "",
        security_alert_email=app.config.get("SECURITY_ALERT_EMAIL") or "",
        webhook_url=app.config.get("ALERT_WEBHOOK_URL") or "",
        webhook_secret=app.config.get("ALERT_WEBHOOK_SECRET") or "",
        max_attempts=int(app.config.get("ALERT_NOTIFY_MAX_ATTEMPTS") or 3),
        enabled=bool(app.config.get("ALERT_NOTIFY_ON_CREATE", True)),
    )
    app.extensions["alert_store"] = store
    app.extensions["notify_config"] = notify_config

    user_store = UserStore(app.config["AUTH_DB_PATH"])
    _seed_env_users(user_store, app.config)
    app.extensions["user_store"] = user_store

    camera_store = CameraStore(app.config["CAMERA_DB_PATH"])
    if app.config.get("SEED_DEMO_CAMERAS", True):
        camera_store.seed_demo_cameras(root)
    app.extensions["camera_store"] = camera_store
    app.extensions["notifier"] = NotificationService(
        store,
        notify_config,
        camera_store=camera_store,
        user_store=user_store,
    )

    from alerts.schema import category_display_name

    from . import routes

    app.register_blueprint(routes.bp)

    @app.template_filter("category_label")
    def _category_label_filter(value):
        return category_display_name(value)

    @app.context_processor
    def inject_globals():
        ckpt = Path(app.config.get("ACTIVITY_CHECKPOINT", ""))
        notify = app.extensions["notify_config"]
        return {
            "safety_banner": "AI detects and alerts. Humans verify and decide.",
            "product_name": "Mun Cyber Eye",
            "company_name": "Mun Cyber Technologies",
            "tagline": "See danger earlier. Alert faster. Protect people.",
            "activity_checkpoint_ready": ckpt.is_file(),
            "phase": 5,
            "notify_resend_configured": notify.resend_configured,
            "notify_webhook_configured": notify.webhook_configured,
            "allow_webcam": bool(app.config.get("ALLOW_WEBCAM")),
            "face_aggression_enabled": _env_flag("ENABLE_FACE_AGGRESSION", "0"),
            "can_train": can_train(session.get("role")),
            "can_approve_accounts": can_approve_accounts(session.get("role")),
        }

    return app
