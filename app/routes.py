"""Flask routes: auth, dashboard, alert review, demo pipeline run."""

from __future__ import annotations

import os
from functools import wraps
from pathlib import Path

from flask import (
    Blueprint,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)
from werkzeug.utils import secure_filename

bp = Blueprint("main", __name__)


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def _store():
    return current_app.extensions["alert_store"]


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if (
            username == current_app.config["ADMIN_USERNAME"]
            and password == current_app.config["ADMIN_PASSWORD"]
        ):
            session["user"] = username
            flash("Signed in. Alerts require human verification.", "ok")
            nxt = request.args.get("next") or url_for("main.dashboard")
            return redirect(nxt)
        flash("Invalid credentials.", "error")
    return render_template("login.html")


@bp.route("/logout")
def logout():
    session.clear()
    flash("Signed out.", "ok")
    return redirect(url_for("main.login"))


@bp.route("/")
@login_required
def dashboard():
    store = _store()
    status = request.args.get("status") or None
    alerts = store.list_alerts(status=status, limit=50)
    stats = store.stats()
    return render_template(
        "dashboard.html",
        alerts=alerts,
        stats=stats,
        filter_status=status or "all",
        vision_backend=current_app.config["VISION_BACKEND"],
    )


@bp.route("/alerts/<alert_id>")
@login_required
def alert_detail(alert_id: str):
    store = _store()
    alert = store.get(alert_id)
    if not alert:
        flash("Alert not found.", "error")
        return redirect(url_for("main.dashboard"))
    audit = store.audit_trail(alert_id)
    return render_template("alert_detail.html", alert=alert, audit=audit)


@bp.route("/alerts/<alert_id>/action", methods=["POST"])
@login_required
def alert_action(alert_id: str):
    action = request.form.get("action", "").strip().lower()
    note = request.form.get("note", "").strip()
    actor = session.get("user", "unknown")
    store = _store()
    try:
        store.apply_action(alert_id, action, actor=actor, note=note)
        flash(f"Alert {action}d by {actor}. Decision recorded in audit log.", "ok")
    except (ValueError, KeyError) as exc:
        flash(str(exc), "error")
    return redirect(url_for("main.alert_detail", alert_id=alert_id))


@bp.route("/run", methods=["GET", "POST"])
@login_required
def run_pipeline():
    """Operator-triggered pipeline run on uploaded or synthetic authorized video."""
    result_summary = None
    if request.method == "POST":
        mode = request.form.get("mode", "synthetic")
        store = _store()
        root = Path(current_app.config["PROJECT_ROOT"])

        # Ensure project root is on path for pipeline imports
        import sys

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from pipeline import CyberEyePipeline, demo_synthetic_run
        from vision.detector import create_adapter

        os.environ["VISION_BACKEND"] = current_app.config["VISION_BACKEND"]
        snapshot_dir = current_app.config["SNAPSHOT_DIR"]

        try:
            if mode == "synthetic":
                result = demo_synthetic_run(store, frames=16)
            else:
                upload = request.files.get("video")
                if not upload or not upload.filename:
                    flash("Select an authorized video file, or use synthetic demo.", "error")
                    return redirect(url_for("main.run_pipeline"))
                safe = secure_filename(upload.filename)
                dest = Path(snapshot_dir).parent / "uploads"
                dest.mkdir(parents=True, exist_ok=True)
                path = dest / safe
                upload.save(path)
                adapter = create_adapter(current_app.config["VISION_BACKEND"])
                pipe = CyberEyePipeline(
                    store=store, adapter=adapter, snapshot_dir=snapshot_dir
                )
                result = pipe.run_video(
                    path,
                    sample_fps=current_app.config["SAMPLE_FPS"],
                    max_frames=current_app.config["MAX_FRAMES_PER_RUN"],
                    source_label=current_app.config["DEFAULT_CAMERA_LABEL"],
                )

            result_summary = {
                "frames": result.frames_processed,
                "alerts": len(result.alerts_created),
                "backend": result.backend,
                "source": result.source_label,
            }
            flash(
                f"Processed {result.frames_processed} frames via {result.backend}; "
                f"{len(result.alerts_created)} alert(s) queued for human review.",
                "ok",
            )
        except Exception as exc:
            flash(f"Pipeline error: {exc}", "error")

    return render_template("run.html", result=result_summary)


@bp.route("/snapshots/<path:filename>")
@login_required
def snapshot_file(filename: str):
    directory = Path(current_app.config["SNAPSHOT_DIR"])
    return send_from_directory(directory, filename)


@bp.route("/health")
def health():
    return {"status": "ok", "product": "Mun Cyber Eye", "phase": 2}
