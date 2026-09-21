"""Flask routes: auth, dashboard, alert review, recipients, demo pipeline run."""

from __future__ import annotations

import json
import os
from collections import Counter
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
from alerts.notify import parse_env_recipients, send_resend_email
from alerts.schema import category_display_name, structured_payload
from vision.dataset import ACTIVITY_CATEGORIES
from vision.face_aggression import face_aggression_enabled
from vision.gunshot_assist import gunshot_audio_enabled
from vision.dangerous_objects import all_dangerous_objects, dangerous_display_name
from vision.objects_catalog import all_objects, object_display_name
from vision.scene_context import all_places, place_display_name, validate_place_type
from vision.sports_catalog import all_sports, sport_display_name
from ingest.cameras import (
    camera_from_form,
    linked_accounts_from_form,
    mask_uri,
    place_type_from_form,
)

from .auth import (
    CUSTOMER_ROLES,
    INACTIVE_LOGIN_MESSAGE,
    LOGIN_CODE_DIGITS,
    LOGIN_CODE_EMAIL_NOTE,
    LOGIN_CODE_MINUTES_DEFAULT,
    LOGIN_CODE_RESEND_SECONDS_DEFAULT,
    PENDING_2FA_NEXT_KEY,
    PENDING_LOGIN_MESSAGE,
    LoginCodeCooldown,
    account_notify_email,
    approver_required,
    begin_login_code_challenge,
    clear_login_code_challenge,
    console_session_guard,
    peek_demo_login_code,
    developer_required,
    guest_only,
    login_required,
    pending_login_code_username,
    public_register_role,
    safe_next_url,
    stash_demo_login_code,
    start_session,
    two_factor_required_for,
    validate_email,
    validate_password,
    validate_username,
)
from . import training as training_ops

bp = Blueprint("main", __name__)

MANAGE_ROLES = {"admin", "operator", "developer"}


def operator_required(view):
    """Admin/operator/developer — recipient management and outbound resend."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        blocked = console_session_guard()
        if blocked is not None:
            return blocked
        if session.get("role") not in MANAGE_ROLES:
            flash("Only admin, operator, or developer roles can manage alert delivery.", "error")
            return redirect(url_for("main.dashboard"))
        return view(*args, **kwargs)

    return wrapped


def _store():
    return current_app.extensions["alert_store"]


def _notifier():
    return current_app.extensions["notifier"]


def _users():
    return current_app.extensions["user_store"]


def _cameras():
    return current_app.extensions["camera_store"]


def _console_users():
    return _users().list_users(active_only=False)


def _session_notify_emails() -> list[str]:
    """Signed-in account alert address (security_email or login email)."""
    username = (session.get("user") or "").strip()
    if not username:
        return []
    user = _users().get_by_username(username)
    if user is None or not user.can_access_console():
        return []
    email = account_notify_email(user)
    return [email] if email else []


def _current_user():
    username = (session.get("user") or "").strip()
    if not username:
        return None
    return _users().get_by_username(username)


def _save_account_cameras(user, form) -> None:
    camera_ids = form.getlist("camera_ids") if hasattr(form, "getlist") else []
    _cameras().set_cameras_for_user(user.id, user.username, camera_ids)
    security = form.get("security_email")
    if security is not None:
        _users().set_security_email(user.id, security)


def _reset_url(token: str) -> str:
    public = (current_app.config.get("PUBLIC_BASE_URL") or "").rstrip("/")
    path = url_for("main.reset_password", token=token)
    if public:
        return f"{public}{path}"
    return url_for("main.reset_password", token=token, _external=True)


def _login_code_email(username: str, code: str, minutes: int) -> tuple[str, str]:
    text = (
        f"Mun Cyber Eye sign-in code\n\n"
        f"Hello {username},\n\n"
        f"Your one-time sign-in code is: {code}\n\n"
        f"This code expires in {minutes} minutes and can be used once.\n"
        "If you did not try to sign in, you can ignore this message.\n"
        "AI detects and alerts. Humans verify and decide.\n"
    )
    html = (
        f"<p>Hello {username},</p>"
        f"<p>Your Mun Cyber Eye one-time sign-in code is:</p>"
        f"<p style=\"font-size:1.4rem;letter-spacing:0.2em\"><strong>{code}</strong></p>"
        f"<p>This code expires in {minutes} minutes and can be used once.</p>"
        f"<p>If you did not try to sign in, you can ignore this message.</p>"
        f"<p>AI detects and alerts. Humans verify and decide.</p>"
    )
    return html, text


def _show_demo_login_code() -> bool:
    return bool(
        current_app.config.get("AUTH_SHOW_LOGIN_CODE") or current_app.testing
    )


def _login_code_ttl_minutes() -> int:
    try:
        return int(current_app.config.get("LOGIN_CODE_MINUTES", LOGIN_CODE_MINUTES_DEFAULT))
    except (TypeError, ValueError):
        return LOGIN_CODE_MINUTES_DEFAULT


def _login_code_resend_seconds() -> int:
    try:
        return int(
            current_app.config.get(
                "LOGIN_CODE_RESEND_SECONDS", LOGIN_CODE_RESEND_SECONDS_DEFAULT
            )
        )
    except (TypeError, ValueError):
        return LOGIN_CODE_RESEND_SECONDS_DEFAULT


def _challenge_user_or_redirect():
    username = pending_login_code_username()
    if session.get("user"):
        blocked = console_session_guard()
        if blocked is not None:
            return None, blocked
        return None, redirect(url_for("main.dashboard"))
    if not username:
        flash("Sign in with your password first.", "error")
        return None, redirect(url_for("main.login"))
    user = _users().get_by_username(username)
    if user is None or not user.can_access_console():
        clear_login_code_challenge()
        if user is not None and not user.approved:
            flash(PENDING_LOGIN_MESSAGE, "error")
        else:
            flash(INACTIVE_LOGIN_MESSAGE, "error")
        return None, redirect(url_for("main.login"))
    if not two_factor_required_for(user):
        return None, _complete_console_login(user)
    return user, None


def _issue_login_code(user, *, resend: bool = False) -> str:
    """Create a hashed code and try Resend. Never claim email was sent on failure.

    Returns ``ok``, ``cooldown``, or ``error``.
    """
    minutes = _login_code_ttl_minutes()
    try:
        code = _users().create_login_code(
            user.id,
            ttl_minutes=minutes,
            min_resend_seconds=_login_code_resend_seconds(),
        )
    except LoginCodeCooldown as exc:
        flash(str(exc), "error")
        return "cooldown"
    except ValueError as exc:
        flash(str(exc), "error")
        return "error"

    current_app.logger.info(
        "Sign-in code for %s (local demo / logs only): %s",
        user.username,
        code,
    )

    notify = current_app.extensions["notify_config"]
    emailed = False
    if notify.resend_configured:
        html, text = _login_code_email(user.username, code, minutes)
        ok, detail = send_resend_email(
            api_key=notify.resend_api_key,
            from_addr=notify.resend_from,
            to=user.email,
            subject="Your Mun Cyber Eye sign-in code",
            html=html,
            text=text,
            user_agent=notify.user_agent,
        )
        if ok:
            emailed = True
            if resend:
                flash(
                    "A new sign-in code was sent to the login email on this account. "
                    "It expires soon and can be used once.",
                    "ok",
                )
            else:
                flash(
                    "A sign-in code was sent to the login email on this account. "
                    "It expires soon and can be used once.",
                    "ok",
                )
        else:
            current_app.logger.error("Sign-in code email failed: %s", detail)
            flash(
                "Email delivery failed. A sign-in code was not emailed. "
                f"{detail}",
                "error",
            )
    else:
        flash(
            "Email delivery is not configured (RESEND_API_KEY). "
            "A sign-in code was not emailed.",
            "error",
        )

    if _show_demo_login_code():
        stash_demo_login_code(code)
        if not emailed:
            flash(
                "Local demo: a one-time sign-in code is shown below and written "
                "to the console log.",
                "ok",
            )
    elif not emailed:
        flash(
            "If this is a local demo, check the application console log for a "
            "one-time sign-in code.",
            "ok",
        )
    return "ok"


def _complete_console_login(user, *, email_2fa_verified: bool = False):
    nxt = session.get(PENDING_2FA_NEXT_KEY) or ""
    start_session(user, email_2fa_verified=email_2fa_verified)
    flash("Signed in. Alerts require human verification.", "ok")
    if nxt.startswith("/") and not nxt.startswith("//"):
        return redirect(nxt)
    return redirect(safe_next_url())


def _password_reset_email(username: str, reset_url: str, minutes: int) -> tuple[str, str]:
    text = (
        f"Mun Cyber Eye password reset\n\n"
        f"Hello {username},\n\n"
        f"A password reset was requested for your operator account. "
        f"This link expires in {minutes} minutes:\n\n{reset_url}\n\n"
        "If you did not request this, you can ignore this message.\n"
        "AI detects and alerts. Humans verify and decide.\n"
    )
    html = (
        f"<p>Hello {username},</p>"
        f"<p>A password reset was requested for your Mun Cyber Eye operator account. "
        f"This link expires in {minutes} minutes.</p>"
        f'<p><a href="{reset_url}">Choose a new password</a></p>'
        f"<p>If you did not request this, you can ignore this message.</p>"
        f"<p>AI detects and alerts. Humans verify and decide.</p>"
    )
    return html, text


@bp.route("/login", methods=["GET", "POST"])
@guest_only
def login():
    needs_setup = _users().count() == 0
    if request.method == "GET" and pending_login_code_username() and not session.get("user"):
        return redirect(url_for("main.login_code"))
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        user, status = _users().attempt_login(username, password)
        if status == "ok" and user:
            if two_factor_required_for(user):
                begin_login_code_challenge(user, safe_next_url())
                issued = _issue_login_code(user)
                if issued == "error":
                    clear_login_code_challenge()
                    return render_template(
                        "login.html",
                        next=request.args.get("next") or request.form.get("next") or "",
                        needs_setup=needs_setup,
                    )
                return redirect(url_for("main.login_code"))
            start_session(user)
            flash("Signed in. Alerts require human verification.", "ok")
            return redirect(safe_next_url())
        if status == "pending":
            flash(PENDING_LOGIN_MESSAGE, "error")
        elif status == "inactive":
            flash(INACTIVE_LOGIN_MESSAGE, "error")
        else:
            flash("Invalid credentials.", "error")
    return render_template(
        "login.html",
        next=request.args.get("next", ""),
        needs_setup=needs_setup,
    )


@bp.route("/register", methods=["GET", "POST"])
@bp.route("/create-account", methods=["GET", "POST"])
@guest_only
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        security_email = request.form.get("security_email", "").strip()
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        err = (
            validate_username(username)
            or validate_email(email)
            or (validate_email(security_email) if security_email else None)
            or validate_password(password, confirm)
        )
        if err:
            flash(err, "error")
            return render_template(
                "register.html", needs_setup=_users().count() == 0
            )
        try:
            is_first = _users().count() == 0
            role = public_register_role(is_first=is_first)
            if role not in CUSTOMER_ROLES:
                raise ValueError("Create account cannot grant developer.")
            user = _users().create_user(
                username=username,
                email=email,
                password=password,
                role=role,
                approved=is_first,
                approved_by="bootstrap" if is_first else "",
            )
            if security_email:
                _users().set_security_email(user.id, security_email)
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template(
                "register.html", needs_setup=_users().count() == 0
            )
        if user.role == "admin":
            flash(
                "First account created as site admin and auto-approved so you can "
                "approve later registrations. Sign in with the password you chose. "
                "No default password is shipped. Site admin runs cameras and review — "
                "not model training.",
                "ok",
            )
        else:
            flash(
                "Account created. It is pending admin approval — you cannot sign in "
                "to cameras, alerts, or Run Pipeline until a site admin or developer "
                "approves it.",
                "ok",
            )
        return redirect(url_for("main.login"))
    return render_template("register.html", needs_setup=_users().count() == 0)


@bp.route("/forgot-password", methods=["GET", "POST"])
@guest_only
def forgot_password():
    demo_reset_url = None
    if request.method == "POST":
        identifier = request.form.get("identifier", "").strip()
        store = _users()
        user = store.find_by_username_or_email(identifier) if identifier else None
        token = None
        reset_url = None
        if user and user.can_access_console():
            minutes = int(current_app.config.get("RESET_TOKEN_MINUTES", 45))
            token = store.create_reset_token(user.id, ttl_minutes=minutes)
            reset_url = _reset_url(token)
            current_app.logger.info(
                "Password reset URL for %s (local demo / logs only): %s",
                user.username,
                reset_url,
            )

        notify = current_app.extensions["notify_config"]
        if notify.resend_configured:
            if user and user.can_access_console() and reset_url:
                minutes = int(current_app.config.get("RESET_TOKEN_MINUTES", 45))
                html, text = _password_reset_email(user.username, reset_url, minutes)
                ok, detail = send_resend_email(
                    api_key=notify.resend_api_key,
                    from_addr=notify.resend_from,
                    to=user.email,
                    subject="Reset your Mun Cyber Eye password",
                    html=html,
                    text=text,
                    user_agent=notify.user_agent,
                )
                if not ok:
                    current_app.logger.error("Password reset email failed: %s", detail)
                    flash(
                        "Email delivery failed. A reset email was not sent. "
                        f"{detail}",
                        "error",
                    )
                    return render_template("forgot_password.html")
            flash(
                "If an account exists for that username or email, password reset "
                "instructions will arrive shortly.",
                "ok",
            )
        else:
            flash(
                "Email delivery is not configured (RESEND_API_KEY). "
                "A reset email was not sent.",
                "error",
            )
            show_demo = current_app.config.get("AUTH_SHOW_RESET_URL") or current_app.testing
            if show_demo and reset_url:
                demo_reset_url = reset_url
                flash(
                    "Local demo: a one-time reset URL is shown below and written to the console log.",
                    "ok",
                )
            elif reset_url:
                flash(
                    "If this is a local demo, check the application console log for a one-time reset URL.",
                    "ok",
                )
    return render_template("forgot_password.html", demo_reset_url=demo_reset_url)


@bp.route("/reset-password", methods=["GET", "POST"])
@guest_only
def reset_password():
    token = (request.values.get("token") or "").strip()
    store = _users()
    user = store.get_by_reset_token(token) if token else None
    if user is None:
        flash("Reset link is invalid or has expired.", "error")
        return redirect(url_for("main.forgot_password"))

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm = request.form.get("confirm_password", "")
        err = validate_password(password, confirm)
        if err:
            flash(err, "error")
            return render_template("reset_password.html", token=token, username=user.username)
        try:
            store.consume_reset_token(token, password)
        except ValueError as exc:
            flash(str(exc), "error")
            return redirect(url_for("main.forgot_password"))
        flash("Password updated. Sign in with your new password.", "ok")
        return redirect(url_for("main.login"))

    return render_template("reset_password.html", token=token, username=user.username)


def _render_login_code(user, *, demo_login_code: str = ""):
    minutes = _login_code_ttl_minutes()
    return render_template(
        "login_code.html",
        username=user.username,
        login_email=user.email,
        minutes=minutes,
        demo_login_code=demo_login_code,
        email_note=LOGIN_CODE_EMAIL_NOTE,
        login_code_digits=LOGIN_CODE_DIGITS,
    )


@bp.route("/login-code", methods=["GET", "POST"])
def login_code():
    user, bounced = _challenge_user_or_redirect()
    if bounced is not None:
        return bounced
    assert user is not None

    if request.method == "POST":
        code = request.form.get("code", "")
        try:
            _users().verify_login_code(user.id, code)
        except ValueError as exc:
            message = str(exc)
            flash(message, "error")
            if "Sign in again" in message or "approved" in message.lower():
                clear_login_code_challenge()
                return redirect(url_for("main.login"))
            return _render_login_code(user, demo_login_code=peek_demo_login_code())
        return _complete_console_login(user, email_2fa_verified=True)

    return _render_login_code(user, demo_login_code=peek_demo_login_code())


@bp.route("/login-code/resend", methods=["POST"])
def resend_login_code():
    user, bounced = _challenge_user_or_redirect()
    if bounced is not None:
        return bounced
    assert user is not None
    _issue_login_code(user, resend=True)
    return redirect(url_for("main.login_code"))


@bp.route("/login-code/cancel", methods=["POST"])
def cancel_login_code():
    username = pending_login_code_username()
    if username:
        found = _users().get_by_username(username)
        if found is not None:
            _users().clear_login_code(found.id)
    clear_login_code_challenge()
    flash("Sign-in code canceled. Enter your password to start again.", "ok")
    return redirect(url_for("main.login"))


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
        cameras=_cameras().list_cameras(),
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
    updated = _notifier().deliver(
        alert,
        actor=actor,
        force=True,
        extra_recipients=_session_notify_emails(),
    )
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


QUIET_RUN_MESSAGE = (
    "Frames were analyzed; no elevated-risk frames under current heuristics / "
    "model (ordinary). That is expected for many real cameras until models "
    "are trained on your site."
)


def _attached_upload():
    """Detect a stray file on Run Pipeline. Uploads are training-only, not detection."""
    upload = request.files.get("video")
    if upload is None:
        return None
    if not (upload.filename or "").strip():
        return None
    return upload


def _pipeline_result_summary(results, *, mode: str = "", filename: str = "") -> dict:
    rows = results if isinstance(results, list) else [results]
    frames = sum(r.frames_processed for r in rows)
    alerts = sum(len(r.alerts_created) for r in rows)
    backends = sorted({r.backend for r in rows if r.backend})
    errors = [r for r in rows if getattr(r, "error", None)]
    counts: Counter[str] = Counter()
    detailed: dict[str, dict] = {}
    for result in rows:
        counts.update(getattr(result, "category_counts", None) or {})
        for assessment in getattr(result, "assessments", []) or []:
            key = assessment.category
            entry = detailed.setdefault(
                key,
                {
                    "category": key,
                    "label": category_display_name(key),
                    "frames": 0,
                    "alerts": 0,
                },
            )
            entry["frames"] += 1
            if assessment.should_alert:
                entry["alerts"] += 1
    if not detailed:
        for key, n in counts.items():
            detailed[key] = {
                "category": key,
                "label": category_display_name(key),
                "frames": n,
                "alerts": 0,
            }
    ordered = [detailed[cat] for cat in ACTIVITY_CATEGORIES if cat in detailed]
    ordered.extend(
        entry for key, entry in detailed.items() if key not in ACTIVITY_CATEGORIES
    )
    sport_counts: Counter[str] = Counter()
    place_counts: Counter[str] = Counter()
    object_counts: Counter[str] = Counter()
    objects_backend = ""
    cue_counts: Counter[str] = Counter()
    max_aggression = 0.0
    max_kit = 0.0
    jersey_frames = 0
    face_statuses: Counter[str] = Counter()
    fall_counts: Counter[str] = Counter()
    gunshot_frames = 0
    aimed_frames = 0
    thrown_frames = 0
    dangerous_counts: Counter[str] = Counter()
    max_weapon_intensity = 0.0
    intensity_counts: Counter[str] = Counter()
    assist_rows = []
    for result in rows:
        for assessment in getattr(result, "assessments", []) or []:
            sport = getattr(assessment, "sport_context", "") or ""
            if sport:
                sport_counts[sport] += 1
            place = getattr(assessment, "place_type", "") or ""
            if place and place != "unknown":
                place_counts[place] += 1
            for obj in getattr(assessment, "objects_seen", []) or []:
                oid = obj.get("id") if isinstance(obj, dict) else str(obj)
                if oid:
                    object_counts[str(oid)] += 1
            backend_name = getattr(assessment, "objects_backend", "") or ""
            if backend_name:
                objects_backend = backend_name
            score = float(getattr(assessment, "aggression_score", 0.0) or 0.0)
            max_aggression = max(max_aggression, score)
            kit = float(getattr(assessment, "team_kit_similarity", 0.0) or 0.0)
            max_kit = max(max_kit, kit)
            if getattr(assessment, "jersey_like_colors", False):
                jersey_frames += 1
            for cue in getattr(assessment, "aggression_cues", []) or []:
                cue_counts[str(cue)] += 1
            face_statuses[getattr(assessment, "face_cue_status", "") or "disabled"] += 1
            manner = getattr(assessment, "fall_manner", "") or ""
            if manner:
                fall_counts[manner] += 1
            if getattr(assessment, "gunshot_proxy", False):
                gunshot_frames += 1
            if getattr(assessment, "aimed_at_person", False):
                aimed_frames += 1
            if getattr(assessment, "thrown_at_person", False):
                thrown_frames += 1
            wid = getattr(assessment, "weapon_id", "") or ""
            if wid:
                dangerous_counts[str(wid)] += 1
            wint = float(getattr(assessment, "weapon_use_intensity", 0.0) or 0.0)
            max_weapon_intensity = max(max_weapon_intensity, wint)
            ilabel = getattr(assessment, "use_intensity_label", "") or ""
            if ilabel and ilabel != "none":
                intensity_counts[str(ilabel)] += 1
            assist_rows.append(
                {
                    "frame": assessment.frame_index,
                    "category": assessment.category,
                    "label": category_display_name(assessment.category),
                    "sport": sport,
                    "sport_label": getattr(assessment, "sport_display", "")
                    or (sport_display_name(sport) if sport else ""),
                    "place": place if place != "unknown" else "",
                    "place_label": getattr(assessment, "place_display", "")
                    or (place_display_name(place) if place and place != "unknown" else ""),
                    "place_source": getattr(assessment, "place_source", "") or "",
                    "kit": round(kit, 3),
                    "jersey": bool(getattr(assessment, "jersey_like_colors", False)),
                    "aggression": round(score, 3),
                    "cues": list(getattr(assessment, "aggression_cues", []) or []),
                    "face": getattr(assessment, "face_cue_status", "disabled"),
                    "alert": bool(assessment.should_alert),
                    "fall_manner": getattr(assessment, "fall_manner", "") or "",
                    "fall_display": getattr(assessment, "fall_display", "") or "",
                    "gunshot": bool(getattr(assessment, "gunshot_proxy", False)),
                    "aimed": bool(getattr(assessment, "aimed_at_person", False)),
                    "thrown": bool(getattr(assessment, "thrown_at_person", False)),
                    "throw_label": getattr(assessment, "throw_label", "") or "",
                    "weapon_tier": getattr(assessment, "weapon_use_tier", "") or "",
                    "weapon_id": getattr(assessment, "weapon_id", "") or "",
                    "weapon_class": getattr(assessment, "weapon_class", "") or "",
                    "harm_potential": getattr(assessment, "harm_potential", "") or "",
                    "use_intensity": round(
                        float(getattr(assessment, "weapon_use_intensity", 0.0) or 0.0), 3
                    ),
                    "use_intensity_label": getattr(assessment, "use_intensity_label", "")
                    or "",
                    "objects": [
                        obj.get("id") if isinstance(obj, dict) else str(obj)
                        for obj in (getattr(assessment, "objects_seen", []) or [])
                    ],
                }
            )
    if not object_counts:
        for result in rows:
            object_counts.update(getattr(result, "object_counts", None) or {})
    if not objects_backend:
        backends_obj = [getattr(r, "objects_backend", "") for r in rows]
        objects_backend = next((b for b in backends_obj if b), "unavailable")
    objects_note = next(
        (getattr(r, "objects_note", "") for r in rows if getattr(r, "objects_note", "")),
        "",
    )
    face_status = "disabled"
    if face_statuses:
        face_status = face_statuses.most_common(1)[0][0]
    return {
        "frames": frames,
        "alerts": alerts,
        "backend": ", ".join(backends) if backends else "",
        "source": rows[0].source_label if len(rows) == 1 else f"{len(rows)} camera(s)",
        "mode": mode,
        "filename": filename,
        "quiet": frames > 0 and alerts == 0,
        "categories": sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
        "category_rows": ordered,
        "sport_contexts": [
            {
                "id": sport,
                "label": sport_display_name(sport),
                "frames": n,
            }
            for sport, n in sport_counts.most_common()
        ],
        "place_types": [
            {
                "id": place,
                "label": place_display_name(place),
                "frames": n,
            }
            for place, n in place_counts.most_common()
        ],
        "objects_backend": objects_backend or "unavailable",
        "objects_note": objects_note,
        "objects": [
            {
                "id": oid,
                "label": object_display_name(oid),
                "count": n,
            }
            for oid, n in object_counts.most_common()
        ],
        "kit": {
            "max_similarity": round(max_kit, 3),
            "jersey_like_frames": jersey_frames,
        },
        "aggression": {
            "max_score": round(max_aggression, 3),
            "cues": [c for c, _n in cue_counts.most_common()],
            "frames_flagged": sum(1 for row in assist_rows if row["cues"]),
        },
        "face_cue_status": face_status,
        "face_enabled": face_aggression_enabled(),
        "fall_manners": [
            {"id": m, "count": n} for m, n in fall_counts.most_common()
        ],
        "gunshot_proxy_frames": gunshot_frames,
        "gunshot_audio_status": next(
            (
                getattr(assessment, "gunshot_audio_status", "")
                for result in rows
                for assessment in getattr(result, "assessments", []) or []
                if getattr(assessment, "gunshot_audio_status", "")
            ),
            "disabled",
        ),
        "aimed_at_person_frames": aimed_frames,
        "thrown_at_person_frames": thrown_frames,
        "dangerous_objects": [
            {
                "id": oid,
                "label": dangerous_display_name(oid),
                "count": n,
            }
            for oid, n in dangerous_counts.most_common()
        ],
        "weapon_use": {
            "max_intensity": round(max_weapon_intensity, 3),
            "labels": [
                {"id": lab, "count": n} for lab, n in intensity_counts.most_common()
            ],
        },
        "assist_rows": assist_rows,
        "cameras": [
            {
                "id": r.camera_id,
                "source": r.source_label,
                "location": r.location_label,
                "frames": r.frames_processed,
                "alerts": len(r.alerts_created),
                "error": r.error,
            }
            for r in rows
        ],
        "failures": len(errors),
    }


def _flash_run_outcome(summary: dict) -> None:
    backend = summary.get("backend") or "unknown"
    frames = summary.get("frames") or 0
    alerts = summary.get("alerts") or 0
    if summary.get("failures"):
        flash(
            f"Processed {frames} frames; "
            f"{alerts} alert(s) queued for human review. "
            f"{summary['failures']} camera(s) reported an honest error "
            "(offline or missing credentials) — no fake detections.",
            "warn",
        )
    else:
        flash(
            f"Processed {frames} frames via {backend}; "
            f"{alerts} alert(s) queued for human review.",
            "ok",
        )
    if summary.get("quiet"):
        flash(QUIET_RUN_MESSAGE, "ok")
        cats = summary.get("categories") or []
        if cats:
            top = ", ".join(f"{label} × {n}" for label, n in cats[:4])
            flash(f"Top predicted categories: {top}.", "ok")


def _run_registered_selection(
    camera_id: str,
    store,
    notifier,
    snapshot_dir: str,
    notify_extra_recipients=None,
):
    from pipeline import CyberEyePipeline, run_registered_cameras
    from vision.detector import create_adapter

    cameras_store = _cameras()
    if camera_id == "all":
        selected = cameras_store.list_cameras(enabled_only=True)
        if not selected:
            raise ValueError("No enabled cameras in the registry.")
    else:
        camera = cameras_store.get(camera_id)
        if camera is None:
            raise ValueError("Camera not found.")
        selected = [camera]

    adapter = create_adapter(
        current_app.config["VISION_BACKEND"],
        checkpoint=current_app.config.get("ACTIVITY_CHECKPOINT"),
    )
    pipe = CyberEyePipeline(
        store=store,
        adapter=adapter,
        snapshot_dir=snapshot_dir,
        notifier=notifier,
        activity_data_root=current_app.config.get("ACTIVITY_DATA_ROOT"),
        notify_extra_recipients=notify_extra_recipients,
    )
    return run_registered_cameras(
        pipe,
        selected,
        cameras_store,
        max_frames=current_app.config["MAX_FRAMES_PER_RUN"],
        project_root=current_app.config["PROJECT_ROOT"],
        allow_webcam=bool(current_app.config.get("ALLOW_WEBCAM")),
        timeout_sec=float(current_app.config.get("RTSP_CONNECT_TIMEOUT_SEC") or 8),
    )


@bp.route("/cameras")
@operator_required
def cameras():
    return render_template(
        "cameras.html",
        cameras=_cameras().list_cameras(),
        camera_accounts=_cameras().accounts_by_camera(),
        allow_webcam=bool(current_app.config.get("ALLOW_WEBCAM")),
        console_users=_console_users(),
    )


@bp.route("/cameras/new", methods=["GET", "POST"])
@operator_required
def camera_new():
    if request.method == "POST":
        try:
            payload = camera_from_form(request.form, user_store=_users())
            camera = _cameras().create(**payload)
            _cameras().set_accounts_for_camera(
                camera.id, linked_accounts_from_form(request.form, _users())
            )
            flash(f"Authorized camera “{camera.name}” registered.", "ok")
            return redirect(url_for("main.cameras"))
        except (ValueError, TypeError) as exc:
            flash(str(exc), "error")
    return render_template(
        "camera_form.html",
        camera=None,
        masked_uri="",
        allow_webcam=bool(current_app.config.get("ALLOW_WEBCAM")),
        place_types=_cameras().list_place_choices(),
        console_users=_console_users(),
        linked_account_ids=set(),
    )


@bp.route("/cameras/<camera_id>/edit", methods=["GET", "POST"])
@operator_required
def camera_edit(camera_id: str):
    store = _cameras()
    camera = store.get(camera_id)
    if camera is None:
        flash("Camera not found.", "error")
        return redirect(url_for("main.cameras"))
    if request.method == "POST":
        try:
            payload = camera_from_form(
                request.form, existing=camera, user_store=_users()
            )
            store.update(camera_id, **payload)
            store.set_accounts_for_camera(
                camera_id, linked_accounts_from_form(request.form, _users())
            )
            flash(f"Camera “{camera.name}” updated.", "ok")
            return redirect(url_for("main.cameras"))
        except (ValueError, TypeError, KeyError) as exc:
            flash(str(exc), "error")
            camera = store.get(camera_id) or camera
    return render_template(
        "camera_form.html",
        camera=camera,
        masked_uri=mask_uri(camera.uri),
        allow_webcam=bool(current_app.config.get("ALLOW_WEBCAM")),
        place_types=_cameras().list_place_choices(),
        console_users=_console_users(),
        linked_account_ids={
            row["user_id"] for row in store.list_accounts_for_camera(camera.id)
        },
    )


@bp.route("/cameras/<camera_id>/enabled", methods=["POST"])
@operator_required
def camera_enabled(camera_id: str):
    enabled = request.form.get("enabled", "0") == "1"
    try:
        camera = _cameras().set_enabled(camera_id, enabled)
        state = "enabled" if camera.enabled else "disabled"
        flash(f"Camera “{camera.name}” {state}.", "ok")
    except KeyError as exc:
        flash(str(exc), "error")
    return redirect(url_for("main.cameras"))


@bp.route("/account", methods=["GET", "POST"])
@login_required
def my_cameras():
    """Account-centric attach/detach for the signed-in user."""
    user = _current_user()
    if user is None:
        return redirect(url_for("main.login"))
    if request.method == "POST":
        try:
            _save_account_cameras(user, request.form)
            flash("Camera assignments and security email saved.", "ok")
            return redirect(url_for("main.my_cameras"))
        except (ValueError, KeyError) as exc:
            flash(str(exc), "error")
        user = _users().get_by_id(user.id) or user
    return render_template(
        "account_cameras.html",
        account=user,
        cameras=_cameras().list_cameras(),
        linked_ids=set(_cameras().list_camera_ids_for_user(user.id)),
        self_edit=True,
    )


@bp.route("/accounts")
@approver_required
def accounts():
    users = _users().list_users(active_only=False)
    pending = [u for u in users if u.access_status() == "pending"]
    counts = {
        user.id: len(_cameras().list_camera_ids_for_user(user.id)) for user in users
    }
    current = _current_user()
    decisions = [
        row
        for row in _store().list_system_audit(limit=40)
        if row.get("action")
        in {
            "approve_account",
            "reject_account",
            "deactivate_account",
            "reactivate_account",
        }
    ][:12]
    return render_template(
        "accounts.html",
        accounts=users,
        pending=pending,
        camera_counts=counts,
        current_user_id=current.id if current else "",
        account_audit=decisions,
    )


@bp.route("/accounts/<user_id>", methods=["GET", "POST"])
@approver_required
def account_edit(user_id: str):
    user = _users().get_by_id(user_id)
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("main.accounts"))
    if request.method == "POST":
        try:
            _save_account_cameras(user, request.form)
            flash(f"Cameras and security email saved for {user.username}.", "ok")
            return redirect(url_for("main.account_edit", user_id=user.id))
        except (ValueError, KeyError) as exc:
            flash(str(exc), "error")
        user = _users().get_by_id(user.id) or user
    return render_template(
        "account_cameras.html",
        account=user,
        cameras=_cameras().list_cameras(),
        linked_ids=set(_cameras().list_camera_ids_for_user(user.id)),
        self_edit=False,
    )


def _account_decision_actor() -> str:
    return session.get("user", "unknown")


def _cannot_change_own_access(user) -> bool:
    actor = (_account_decision_actor() or "").strip().lower()
    return user.username.strip().lower() == actor


@bp.route("/accounts/<user_id>/approve", methods=["POST"])
@approver_required
def account_approve(user_id: str):
    store = _users()
    user = store.get_by_id(user_id)
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("main.accounts"))
    actor = _account_decision_actor()
    updated = store.approve_user(user.id, actor)
    _store().record_system_audit(
        "approve_account",
        actor,
        f"user={updated.username} role={updated.role}",
    )
    flash(
        f"Approved {updated.username}. They can now sign in to cameras, alerts, and Run Pipeline.",
        "ok",
    )
    return redirect(url_for("main.accounts"))


@bp.route("/accounts/<user_id>/reject", methods=["POST"])
@approver_required
def account_reject(user_id: str):
    store = _users()
    user = store.get_by_id(user_id)
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("main.accounts"))
    if _cannot_change_own_access(user):
        flash("You cannot reject your own account.", "error")
        return redirect(url_for("main.accounts"))
    actor = _account_decision_actor()
    try:
        updated = store.reject_user(user.id, actor)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.accounts"))
    _store().record_system_audit(
        "reject_account",
        actor,
        f"user={updated.username} role={updated.role}",
    )
    flash(
        f"Rejected {updated.username}. They cannot sign in until approved again.",
        "ok",
    )
    return redirect(url_for("main.accounts"))


@bp.route("/accounts/<user_id>/active", methods=["POST"])
@approver_required
def account_active(user_id: str):
    store = _users()
    user = store.get_by_id(user_id)
    if user is None:
        flash("Account not found.", "error")
        return redirect(url_for("main.accounts"))
    active = request.form.get("active", "0") == "1"
    if not active and _cannot_change_own_access(user):
        flash("You cannot deactivate your own account.", "error")
        return redirect(url_for("main.accounts"))
    actor = _account_decision_actor()
    try:
        updated = store.set_user_active(user.id, active)
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.accounts"))
    action = "reactivate_account" if updated.active else "deactivate_account"
    _store().record_system_audit(
        action,
        actor,
        f"user={updated.username} role={updated.role}",
    )
    state = "reactivated" if updated.active else "deactivated"
    flash(f"{updated.username} {state}.", "ok")
    return redirect(url_for("main.accounts"))


@bp.route("/run", methods=["GET", "POST"])
@login_required
def run_pipeline():
    """Operator-triggered run on registered cameras (product path) or lab MOCK."""
    result_summary = None
    cameras = _cameras().list_cameras()
    if request.method == "POST":
        store = _store()
        notifier = _notifier()
        actor_emails = _session_notify_emails()
        root = Path(current_app.config["PROJECT_ROOT"])

        import sys

        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from pipeline import demo_activity_run, demo_synthetic_run

        os.environ["VISION_BACKEND"] = current_app.config["VISION_BACKEND"]
        os.environ["ACTIVITY_CHECKPOINT"] = current_app.config["ACTIVITY_CHECKPOINT"]
        snapshot_dir = current_app.config["SNAPSHOT_DIR"]

        if _attached_upload() is not None:
            flash(
                "Video file uploads are for training only "
                "(Train models → Extract frames from video). "
                "This run ignored the attached file. Detection uses registered cameras.",
                "warn",
            )

        mode = (request.form.get("mode") or "camera").strip() or "camera"

        try:
            if mode == "upload":
                flash(
                    "Authorized video file upload is not a customer detection path. "
                    "Use Train models to extract labeled frames, or run registered cameras.",
                    "error",
                )
                return redirect(url_for("main.run_pipeline"))
            if mode == "synthetic":
                result = demo_synthetic_run(
                    store,
                    frames=16,
                    notifier=notifier,
                    snapshot_dir=snapshot_dir,
                    notify_extra_recipients=actor_emails,
                )
                result_summary = _pipeline_result_summary(result, mode="synthetic")
            elif mode == "activity":
                result = demo_activity_run(
                    store,
                    checkpoint=current_app.config["ACTIVITY_CHECKPOINT"],
                    notifier=notifier,
                    snapshot_dir=snapshot_dir,
                    notify_extra_recipients=actor_emails,
                )
                result_summary = _pipeline_result_summary(result, mode="activity")
            elif mode == "camera":
                camera_id = (request.form.get("camera_id") or "").strip()
                if not camera_id:
                    flash("Select a registered camera, or all enabled cameras.", "error")
                    return redirect(url_for("main.run_pipeline"))
                results = _run_registered_selection(
                    camera_id,
                    store,
                    notifier,
                    snapshot_dir,
                    notify_extra_recipients=actor_emails,
                )
                result_summary = _pipeline_result_summary(results, mode="camera")
            else:
                flash(
                    "Select registered cameras (product path) or a lab-only MOCK demo.",
                    "error",
                )
                return redirect(url_for("main.run_pipeline"))

            _flash_run_outcome(result_summary)
        except Exception as exc:
            flash(f"Pipeline error: {exc}", "error")

    return render_template(
        "run.html",
        result=result_summary,
        cameras=cameras,
        enabled_cameras=[c for c in cameras if c.enabled],
    )


def _train_page_context(extra: dict | None = None) -> dict:
    from vision.dataset import ACTIVITY_CATEGORIES, SPLITS
    from vision.sports_catalog import all_sports

    ctx = {
        "checkpoint": training_ops.current_checkpoint_info(current_app.config),
        "dataset": training_ops.dataset_inventory(current_app.config),
        "categories": list(ACTIVITY_CATEGORIES),
        "sports": all_sports(),
        "places": _cameras().list_place_choices(),
        "objects": all_objects(),
        "dangerous_objects": all_dangerous_objects(),
        "object_groups": {
            "home": [o for o in all_objects() if o.group == "home"],
            "community": [o for o in all_objects() if o.group == "community"],
        },
        "objects_dataset": training_ops.objects_inventory(current_app.config),
        "dangerous_dataset": training_ops.dangerous_inventory(current_app.config),
        "splits": list(SPLITS),
        "checkpoints": training_ops.list_checkpoint_files(current_app.config),
        "audit": _store().list_system_audit(limit=20),
        "last_metrics": None,
        "last_eval": None,
        "metric_reports": {},
    }
    if extra:
        ctx.update(extra)
    return ctx


@bp.route("/train")
@developer_required
def train_redirect():
    return redirect(url_for("main.admin_train"))


@bp.route("/admin/train", methods=["GET"])
@developer_required
def admin_train():
    """Mun Cyber developer only — customers do not see this page."""
    return render_template("train.html", **_train_page_context())


@bp.route("/admin/train/run", methods=["POST"])
@developer_required
def admin_train_run():
    actor = session.get("user", "unknown")
    try:
        n_train = int(request.form.get("n_train") or (8 if current_app.testing else 40))
        n_val = int(request.form.get("n_val") or (4 if current_app.testing else 12))
        n_test = int(request.form.get("n_test") or (4 if current_app.testing else 12))
        dest, metrics, bundle = training_ops.run_training(
            current_app.config,
            model_type=(request.form.get("model_type") or "forest").strip().lower(),
            output_name=request.form.get("output_name") or "activity_custom.joblib",
            generate_demo=request.form.get("generate_demo") == "1",
            overwrite_demo_images=request.form.get("overwrite_demo_images") == "1",
            confirm_overwrite_demo=request.form.get("confirm_overwrite_demo") == "1",
            seed=int(request.form.get("seed") or 7),
            n_train=max(1, n_train),
            n_val=max(0, n_val),
            n_test=max(0, n_test),
        )
    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    except Exception as exc:
        flash(f"Training failed: {exc}", "error")
        return redirect(url_for("main.admin_train"))

    acc = None
    if isinstance(metrics.get("test"), dict):
        acc = metrics["test"].get("accuracy")
    elif isinstance(metrics.get("val"), dict):
        acc = metrics["val"].get("accuracy")
    note = f"checkpoint={dest} model={bundle.model_type}"
    if acc is not None:
        note += f" accuracy={acc}"
    _store().record_system_audit("train_model", actor, note)
    flash(
        f"Training finished in-request. Wrote {dest.name}. "
        "Metrics are stored in the checkpoint. Humans still verify every alert.",
        "ok",
    )
    return render_template(
        "train.html",
        **_train_page_context(
            {
                "last_metrics": metrics,
                "metric_reports": training_ops.metrics_as_text(metrics),
            }
        ),
    )


@bp.route("/admin/train/activate", methods=["POST"])
@developer_required
def admin_train_activate():
    actor = session.get("user", "unknown")
    chosen = (request.form.get("checkpoint_path") or "").strip()
    try:
        dest = training_ops.apply_active_checkpoint(current_app.config, chosen, actor)
    except (ValueError, OSError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    _store().record_system_audit(
        "activate_checkpoint", actor, f"path={dest}"
    )
    flash(
        f"Active checkpoint is now {dest}. "
        "Registered-camera runs use this file. "
        "Pointer: data/active_checkpoint.json (not a secret).",
        "ok",
    )
    return redirect(url_for("main.admin_train"))


@bp.route("/admin/train/evaluate", methods=["POST"])
@developer_required
def admin_train_evaluate():
    actor = session.get("user", "unknown")
    split = (request.form.get("split") or "test").strip().lower()
    chosen = (request.form.get("checkpoint_path") or "").strip() or None
    try:
        report = training_ops.run_evaluation(
            current_app.config, split=split, checkpoint=chosen
        )
    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    _store().record_system_audit(
        "evaluate_checkpoint",
        actor,
        f"split={split} accuracy={report.get('accuracy')} n={report.get('n_samples')}",
    )
    flash(f"Evaluation on {split}: accuracy {report.get('accuracy')}.", "ok")
    from vision.metrics import format_metrics_report

    return render_template(
        "train.html",
        **_train_page_context(
            {
                "last_eval": report,
                "metric_reports": {split: format_metrics_report(report)},
            }
        ),
    )


@bp.route("/admin/train/upload", methods=["POST"])
@developer_required
def admin_train_upload():
    actor = session.get("user", "unknown")
    category = request.form.get("category") or ""
    split = request.form.get("split") or "train"
    sport_context = (request.form.get("sport_context") or "").strip()
    place_type = place_type_from_form(request.form)
    zip_file = request.files.get("zipfile")
    images = request.files.getlist("images")
    try:
        folder = (
            training_ops.folder_label_for_upload(category, sport_context, place_type)
            if category or sport_context or place_type
            else ""
        )
        if zip_file and zip_file.filename:
            saved = training_ops.save_labeled_zip(
                current_app.config,
                zip_file,
                default_category=folder or None,
                default_split=split,
            )
            kind = "zip"
        else:
            saved = training_ops.save_labeled_files(
                current_app.config,
                images,
                category=folder or category,
                split=split,
                sport_context="" if folder else sport_context,
                place_type="" if folder else place_type,
            )
            kind = "files"
    except ValueError as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    slug = validate_place_type(place_type)
    if slug:
        _cameras().remember_custom_place(slug, place_type)
    _store().record_system_audit(
        "upload_labels",
        actor,
        f"{kind} count={saved} split={split} category={folder or category or 'from-zip'} place={place_type or '-'}",
    )
    flash(
        f"Saved {saved} labeled frame(s). Training improves assistive detection only.",
        "ok",
    )
    return redirect(url_for("main.admin_train"))


@bp.route("/admin/train/extract-video", methods=["POST"])
@developer_required
def admin_train_extract_video():
    actor = session.get("user", "unknown")
    video = request.files.get("video")
    try:
        fps = float(request.form.get("sample_fps") or current_app.config.get("SAMPLE_FPS") or 2)
        max_frames = int(request.form.get("max_frames") or 60)
        extract_place = place_type_from_form(request.form)
        result = training_ops.extract_video_frames(
            current_app.config,
            video,
            kind=(request.form.get("kind") or "activity").strip().lower(),
            split=(request.form.get("split") or "train").strip().lower(),
            category=request.form.get("category") or "",
            sport_context=(request.form.get("sport_context") or "").strip(),
            place_type=extract_place,
            object_id=(
                request.form.get("object_id") or request.form.get("dangerous_id") or ""
            ).strip(),
            sample_fps=fps,
            max_frames=max_frames,
        )
    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    slug = validate_place_type(extract_place)
    if slug:
        _cameras().remember_custom_place(slug, extract_place)
    _store().record_system_audit(
        "extract_video_frames",
        actor,
        (
            f"source={result['source']} frames={result['frames']} "
            f"kind={result['kind']} folder={result['folder']} split={result['split']}"
        ),
    )
    flash(
        f"Sampled {result['frames']} frame(s) from {result['source']} into "
        f"{result['folder']} ({result['split']}). Videos are sampled to frames; "
        "the sklearn activity trainer still learns from images. Humans verify.",
        "ok",
    )
    return redirect(url_for("main.admin_train"))


@bp.route("/admin/train/objects", methods=["POST"])
@developer_required
def admin_train_objects():
    actor = session.get("user", "unknown")
    try:
        dest, metrics, bundle = training_ops.run_object_training(
            current_app.config,
            model_type=(request.form.get("model_type") or "forest").strip().lower(),
            output_name=request.form.get("output_name") or "objects_custom.joblib",
            seed=int(request.form.get("seed") or 7),
        )
    except (ValueError, TypeError) as exc:
        flash(str(exc), "error")
        return redirect(url_for("main.admin_train"))
    except Exception as exc:
        flash(f"Object training failed: {exc}", "error")
        return redirect(url_for("main.admin_train"))
    classes = bundle.get("categories") or []
    _store().record_system_audit(
        "train_object_model",
        actor,
        f"checkpoint={dest} classes={len(classes)}",
    )
    flash(
        f"Object classifier wrote {dest.name} ({len(classes)} class(es)). "
        "Runtime inventory still prefers YOLO and will not invent objects "
        "when the detector is missing. Humans verify.",
        "ok",
    )
    return render_template(
        "train.html",
        **_train_page_context(
            {
                "last_metrics": metrics,
                "metric_reports": training_ops.metrics_as_text(metrics),
            }
        ),
    )


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
        "phase": 5,
        "vision_backend": current_app.config.get("VISION_BACKEND"),
        "activity_checkpoint_ready": ckpt.is_file(),
        "activity_checkpoint": str(ckpt) if ckpt else "",
        "allow_webcam": bool(current_app.config.get("ALLOW_WEBCAM")),
        "face_aggression_enabled": face_aggression_enabled(),
        "gunshot_audio_enabled": gunshot_audio_enabled(),
        "sports_catalog_size": len(all_sports()),
        "place_catalog_size": len(all_places()),
        "objects_catalog_size": len(all_objects()),
        "dangerous_objects_catalog_size": len(all_dangerous_objects()),
        "notify": {
            "resend_configured": notify.resend_configured,
            "webhook_configured": notify.webhook_configured,
            "recipient_count": len(_notifier().recipient_emails()),
            "db_operators": len(store.list_recipients(active_only=True)),
        },
        "cameras": {
            "total": _cameras().count(),
            "enabled": len(_cameras().list_cameras(enabled_only=True)),
        },
    }
