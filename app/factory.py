"""Flask application factory for Mun Cyber Eye console."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from alerts.notify import NotificationService, NotifyConfig
from alerts.store import AlertStore
from ingest.cameras import CameraStore

from .auth import UserStore


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _seed_env_users(user_store: UserStore, config: dict) -> None:
    sync = bool(config.get("ADMIN_SYNC_PASSWORD", True))
    admin_username = config["ADMIN_USERNAME"]
    user_store.ensure_bootstrap_admin(
        admin_username,
        config["ADMIN_PASSWORD"],
        config.get("ADMIN_EMAIL") or f"{admin_username}@localhost",
        sync_password=sync,
        role=config.get("ADMIN_ROLE") or "admin",
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


def create_app(test_config: dict | None = None) -> Flask:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    admin_username = os.getenv("ADMIN_USERNAME", "operator")
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-only-change-me"),
        ADMIN_USERNAME=admin_username,
        ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD", "changeme"),
        ADMIN_ROLE=os.getenv("ADMIN_ROLE", "admin"),
        ADMIN_EMAIL=os.getenv("ADMIN_EMAIL", f"{admin_username}@localhost"),
        ADMIN_SYNC_PASSWORD=_env_flag("ADMIN_SYNC_PASSWORD", "1"),
        OPERATOR_USERNAME=os.getenv("OPERATOR_USERNAME", ""),
        OPERATOR_PASSWORD=os.getenv("OPERATOR_PASSWORD", ""),
        OPERATOR_EMAIL=os.getenv("OPERATOR_EMAIL", ""),
        ALERT_DB_PATH=os.getenv("ALERT_DB_PATH", str(root / "data" / "alerts.db")),
        AUTH_DB_PATH=os.getenv("AUTH_DB_PATH", str(root / "data" / "auth.db")),
        CAMERA_DB_PATH=os.getenv("CAMERA_DB_PATH", str(root / "data" / "cameras.db")),
        SNAPSHOT_DIR=os.getenv("SNAPSHOT_DIR", str(root / "data" / "snapshots")),
        ALLOW_WEBCAM=_env_flag("ALLOW_WEBCAM", "0"),
        RTSP_CONNECT_TIMEOUT_SEC=float(os.getenv("RTSP_CONNECT_TIMEOUT_SEC", "8")),
        VISION_BACKEND=os.getenv("VISION_BACKEND", "auto"),
        ACTIVITY_CHECKPOINT=os.getenv(
            "ACTIVITY_CHECKPOINT",
            str(root / "data" / "checkpoints" / "activity_demo.joblib"),
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
        ALERT_WEBHOOK_URL=os.getenv("ALERT_WEBHOOK_URL", ""),
        ALERT_WEBHOOK_SECRET=os.getenv("ALERT_WEBHOOK_SECRET", ""),
        ALERT_NOTIFY_ON_CREATE=os.getenv("ALERT_NOTIFY_ON_CREATE", "1") != "0",
        ALERT_NOTIFY_MAX_ATTEMPTS=int(os.getenv("ALERT_NOTIFY_MAX_ATTEMPTS", "3")),
        PROJECT_ROOT=str(root),
        PUBLIC_BASE_URL=os.getenv("PUBLIC_BASE_URL", "").rstrip("/"),
        RESET_TOKEN_MINUTES=int(os.getenv("RESET_TOKEN_MINUTES", "45")),
        AUTH_SHOW_RESET_URL=_env_flag("AUTH_SHOW_RESET_URL", "0"),
    )
    if test_config:
        app.config.update(test_config)
        if "CAMERA_DB_PATH" not in test_config and test_config.get("ALERT_DB_PATH"):
            app.config["CAMERA_DB_PATH"] = str(
                Path(app.config["ALERT_DB_PATH"]).with_name("cameras.db")
            )

    def _abs(path_value: str) -> str:
        path = Path(path_value)
        return str(path if path.is_absolute() else root / path)

    app.config["ALERT_DB_PATH"] = _abs(app.config["ALERT_DB_PATH"])
    app.config["AUTH_DB_PATH"] = _abs(app.config["AUTH_DB_PATH"])
    app.config["CAMERA_DB_PATH"] = _abs(
        app.config.get("CAMERA_DB_PATH") or str(root / "data" / "cameras.db")
    )
    app.config["SNAPSHOT_DIR"] = _abs(app.config["SNAPSHOT_DIR"])
    app.config["ACTIVITY_CHECKPOINT"] = _abs(app.config["ACTIVITY_CHECKPOINT"])
    app.config["ALLOW_WEBCAM"] = bool(app.config.get("ALLOW_WEBCAM", False))
    try:
        app.config["RTSP_CONNECT_TIMEOUT_SEC"] = float(
            app.config.get("RTSP_CONNECT_TIMEOUT_SEC", 8)
        )
    except (TypeError, ValueError):
        app.config["RTSP_CONNECT_TIMEOUT_SEC"] = 8.0

    Path(app.config["ALERT_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["AUTH_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["CAMERA_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["SNAPSHOT_DIR"]).mkdir(parents=True, exist_ok=True)

    store = AlertStore(app.config["ALERT_DB_PATH"])
    notify_config = NotifyConfig(
        resend_api_key=app.config.get("RESEND_API_KEY") or "",
        resend_from=app.config.get("RESEND_FROM")
        or "Mun Cyber Technologies <info@muncyber.com>",
        email_recipients_env=app.config.get("ALERT_EMAIL_RECIPIENTS") or "",
        webhook_url=app.config.get("ALERT_WEBHOOK_URL") or "",
        webhook_secret=app.config.get("ALERT_WEBHOOK_SECRET") or "",
        max_attempts=int(app.config.get("ALERT_NOTIFY_MAX_ATTEMPTS") or 3),
        enabled=bool(app.config.get("ALERT_NOTIFY_ON_CREATE", True)),
    )
    app.extensions["alert_store"] = store
    app.extensions["notify_config"] = notify_config
    app.extensions["notifier"] = NotificationService(store, notify_config)

    user_store = UserStore(app.config["AUTH_DB_PATH"])
    _seed_env_users(user_store, app.config)
    app.extensions["user_store"] = user_store

    camera_store = CameraStore(app.config["CAMERA_DB_PATH"])
    if app.config.get("SEED_DEMO_CAMERAS", True):
        camera_store.seed_demo_cameras(root)
    app.extensions["camera_store"] = camera_store

    from . import routes

    app.register_blueprint(routes.bp)

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
        }

    return app
