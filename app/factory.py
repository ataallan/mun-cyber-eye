"""Flask application factory for Mun Cyber Eye console."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

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
        ALERT_DB_PATH=os.getenv("ALERT_DB_PATH", str(root / "data" / "alerts.db")),
        SNAPSHOT_DIR=os.getenv("SNAPSHOT_DIR", str(root / "data" / "snapshots")),
        VISION_BACKEND=os.getenv("VISION_BACKEND", "auto"),
        SAMPLE_FPS=float(os.getenv("SAMPLE_FPS", "2")),
        MAX_FRAMES_PER_RUN=int(os.getenv("MAX_FRAMES_PER_RUN", "120")),
        DEFAULT_CAMERA_LABEL=os.getenv(
            "DEFAULT_CAMERA_LABEL", "Authorized Camera — Demo Lab"
        ),
        PROJECT_ROOT=str(root),
    )
    if test_config:
        app.config.update(test_config)

    Path(app.config["ALERT_DB_PATH"]).parent.mkdir(parents=True, exist_ok=True)
    Path(app.config["SNAPSHOT_DIR"]).mkdir(parents=True, exist_ok=True)

    app.extensions["alert_store"] = AlertStore(app.config["ALERT_DB_PATH"])

    from . import routes

    app.register_blueprint(routes.bp)

    @app.context_processor
    def inject_globals():
        return {
            "safety_banner": "AI detects and alerts. Humans verify and decide.",
            "product_name": "Mun Cyber Eye",
            "company_name": "Mun Cyber Technologies",
            "tagline": "See danger earlier. Alert faster. Protect people.",
        }

    return app
