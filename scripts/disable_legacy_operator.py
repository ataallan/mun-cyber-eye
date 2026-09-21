#!/usr/bin/env python3
"""Deactivate the Capstone demo ``operator`` account when it is still a seed.

Run from the product root (or from a standalone unzip):

    python scripts/disable_legacy_operator.py

The console also does this on startup when ``DISABLE_LEGACY_OPERATOR=1``
(the product default). This script is the one-shot equivalent for an old
database without waiting for a restart.

What it deactivates
    Username ``operator`` **and** either:
    - the password still matches the published demo ``changeme``, or
    - the email is the legacy seed ``operator@localhost``

What it leaves alone
    Username ``operator`` with a strong password and a non-legacy email
    (a customer who chose that username on purpose).

Sign-in as ``operator`` / ``changeme`` is **always** rejected by the console,
even if this script is not run and even if the stored hash still matches.

If deactivating this row leaves you without an admin, set ``ADMIN_USERNAME``
and ``ADMIN_PASSWORD`` in ``.env`` to a strong password you choose, then
restart. Do not use operator/changeme.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.auth import UserStore, disable_legacy_operator  # noqa: E402


def _abs(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    auth_path = _abs(os.getenv("AUTH_DB_PATH", "data/auth.db"))
    if not auth_path.is_file():
        print(f"No auth database at {auth_path}. Nothing to do.")
        return 0
    store = UserStore(auth_path)
    result = disable_legacy_operator(store)
    status = result.get("status")
    note = result.get("note") or ""
    print(f"{auth_path}: {status}")
    if note:
        print(note)
    if status == "deactivated":
        alert_path = _abs(os.getenv("ALERT_DB_PATH", "data/alerts.db"))
        try:
            from alerts.store import AlertStore

            AlertStore(alert_path).record_system_audit(
                "disable_legacy_operator",
                "script",
                note,
            )
            print(f"Wrote system_audit to {alert_path}")
        except Exception as exc:  # pragma: no cover - audit is best-effort
            print(f"Could not write system_audit ({exc})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
