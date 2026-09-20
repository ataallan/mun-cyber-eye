"""Flask application factory for Mun Cyber Eye console."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from alerts.notify import NotificationService, NotifyConfig
from alerts.store import AlertStore


def create_app(test_config: dict | None = None) -> Flask:
    root = Path(__file__).resolve().parent.parent
    load_dotenv(root / ".env")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.update(
        SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-only-change-me"),
        ADMIN_USERNAME=os.getenv("ADMIN_USERNAME", "operator"),
        ADMIN_PASSWORD=os.getenv("ADMIN_PASSWORD", "changeme"),
        ADMIN_ROLE=os.getenv("ADMIN_ROLE", "admin"),
        OPERATOR_USERNAME=os.getenv("OPERATOR_USERNAME", ""),
        OPERATOR_PASSWORD=os.getenv("OPERATOR_PASSWORD", ""),
        ALERT_DB_PATH=os.getenv("ALERT_DB_PATH", str(root / "data" / "alerts.db")),
        SNAPSHOT_DIR=os.getenv("SNAPSHOT_DIR", str(root / "data" / "snapshots")),
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
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["ALERT_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
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
            "phase": 4,
            "notify_resend_configured": notify.resend_configured,
            "notify_webhook_configured": notify.webhook_configured,
        }

    return app
