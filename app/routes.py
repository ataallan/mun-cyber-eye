"""Flask routes: auth, dashboard, alert review, recipients, demo pipeline run."""

from __future__ import annotations

import json
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

from alerts.notify import parse_env_recipients
from alerts.schema import structured_payload

bp = Blueprint("main", __name__)

MANAGE_ROLES = {"admin", "operator"}


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def operator_required(view):
    """Admin/operator only — recipient management and outbound resend."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user"):
            return redirect(url_for("main.login", next=request.path))
        if session.get("role") not in MANAGE_ROLES:
            flash("Only admin or operator roles can manage alert delivery.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def _store():
    return current_app.extensions["alert_store"]


def _notifier():
    return current_app.extensions["notifier"]


def _auth_users() -> dict[str, tuple[str, str]]:
    users: dict[str, tuple[str, str]] = {}
    admin_user = current_app.config["ADMIN_USERNAME"]
    admin_pass = current_app.config["ADMIN_PASSWORD"]
    admin_role = current_app.config.get("ADMIN_ROLE") or "admin"
    users[admin_user] = (admin_pass, admin_role)
    op_user = (current_app.config.get("OPERATOR_USERNAME") or "").strip()
    op_pass = current_app.config.get("OPERATOR_PASSWORD") or ""
    if op_user and op_pass:
        users[op_user] = (op_pass, "operator")
    return users


@bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = _auth_users()
        if username in users and password == users[username][0]:
            session["user"] = username
            session["role"] = users[username][1]
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
        delivery_stats=store.delivery_stats(),
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
    delivery = store.delivery_trail(alert_id)
    payload = structured_payload(alert)
    can_resend = session.get("role") in MANAGE_ROLES
    return render_template(
        "alert_detail.html",
        alert=alert,
        audit=audit,
        delivery=delivery,
        payload=payload,
        payload_json=json.dumps(payload, indent=2, default=str),
        can_resend=can_resend,
    )


@bp.route("/alerts/<alert_id>.json")
@login_required
def alert_json(alert_id: str):
    alert = _store().get(alert_id)
    if not alert:
        return {"error": "not_found"}, 404
    return structured_payload(alert)


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


@bp.route("/alerts/<alert_id>/notify", methods=["POST"])
@operator_required
def alert_resend(alert_id: str):
    store = _store()
    alert = store.get(alert_id)
    if not alert:
        flash("Alert not found.", "error")
        return redirect(url_for("main.dashboard"))
    actor = session.get("user", "unknown")
    updated = _notifier().deliver(alert, actor=actor, force=True)
    if updated.delivery_status in {"sent", "partial"}:
        flash_cat = "ok"
    elif updated.delivery_status in {"queued", "undelivered"}:
        flash_cat = "warn"
    else:
        flash_cat = "error"
    flash(
        f"Delivery attempt recorded as {updated.delivery_status}. "
        "Success is only shown when a channel actually accepted the message.",
        flash_cat,
    )
    return redirect(url_for("main.alert_detail", alert_id=alert_id))


@bp.route("/recipients", methods=["GET", "POST"])
@operator_required
def recipients():
    store = _store()
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        display_name = request.form.get("display_name", "").strip()
        role = request.form.get("role", "operator").strip().lower()
        actor = session.get("user", "unknown")
        try:
            store.add_recipient(
                email, display_name=display_name, role=role, created_by=actor
            )
            flash(f"Authorized recipient {email} saved.", "ok")
        except ValueError as exc:
            flash(str(exc), "error")
        return redirect(url_for("main.recipients"))

    env_recipients = parse_env_recipients(
        current_app.config.get("ALERT_EMAIL_RECIPIENTS") or ""
    )
    return render_template(
        "recipients.html",
        recipients=store.list_recipients(),
        env_recipients=env_recipients,
        notify=_notifier().config,
    )


@bp.route("/recipients/<int:recipient_id>/active", methods=["POST"])
@operator_required
def recipient_active(recipient_id: int):
    active = request.form.get("active", "0") == "1"
    try:
        row = _store().set_recipient_active(recipient_id, active)
        state = "activated" if row["active"] else "deactivated"
        flash(f"Recipient {row['email']} {state}.", "ok")
    except KeyError as exc:
        flash(str(exc), "error")
    return redirect(url_for("main.recipients"))


@bp.route("/run", methods=["GET", "POST"])
@login_required
def run_pipeline():
    """Operator-triggered pipeline run on uploaded or synthetic authorized video."""
    result_summary = None
    if request.method == "POST":
        mode = request.form.get("mode", "synthetic")
        store = _store()
        notifier = _notifier()
        root = Path(current_app.config["PROJECT_ROOT"])

        import sys

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from pipeline import CyberEyePipeline, demo_activity_run, demo_synthetic_run
        from vision.detector import create_adapter

        os.environ["VISION_BACKEND"] = current_app.config["VISION_BACKEND"]
        os.environ["ACTIVITY_CHECKPOINT"] = current_app.config["ACTIVITY_CHECKPOINT"]
        snapshot_dir = current_app.config["SNAPSHOT_DIR"]

        try:
            if mode == "synthetic":
                result = demo_synthetic_run(
                    store,
                    frames=16,
                    notifier=notifier,
                    snapshot_dir=snapshot_dir,
                )
            elif mode == "activity":
                result = demo_activity_run(
                    store,
                    checkpoint=current_app.config["ACTIVITY_CHECKPOINT"],
                    notifier=notifier,
                    snapshot_dir=snapshot_dir,
                )
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
                    store=store,
                    adapter=adapter,
                    snapshot_dir=snapshot_dir,
                    notifier=notifier,
                    location_label=current_app.config.get("DEFAULT_LOCATION_LABEL"),
                    camera_id=current_app.config.get("DEFAULT_CAMERA_ID"),
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
    ckpt = Path(current_app.config.get("ACTIVITY_CHECKPOINT", ""))
    notify = current_app.extensions["notify_config"]
    store = _store()
    return {
        "status": "ok",
        "product": "Mun Cyber Eye",
        "phase": 4,
        "vision_backend": current_app.config.get("VISION_BACKEND"),
        "activity_checkpoint_ready": ckpt.is_file(),
        "notify": {
            "resend_configured": notify.resend_configured,
            "webhook_configured": notify.webhook_configured,
            "recipient_count": len(_notifier().recipient_emails()),
            "db_operators": len(store.list_recipients(active_only=True)),
        },
    }
