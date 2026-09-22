#!/usr/bin/env python3
"""Create a customer .env from .env.example and fill FLASK_SECRET_KEY.

The installed app and the Desktop launcher call this before run.py.
A real .env is left in place. Placeholder secrets are replaced.
The generated key is never printed.
"""

from __future__ import annotations

import secrets
import sys
from pathlib import Path

PLACEHOLDER_SECRETS = frozenset(
    {
        "",
        "change-me",
        "change-me-to-a-long-random-string",
        "changeme",
        "dev-only-change-me",
    }
)


def default_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _assignment(line: str) -> tuple[str, str] | None:
    body = line.strip()
    if not body or body.startswith("#") or "=" not in body:
        return None
    key, value = body.split("=", 1)
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def _replace_placeholder_secret(text: str) -> tuple[str, bool]:
    lines = text.splitlines(keepends=True)
    changed = False
    found = False
    rewritten: list[str] = []
    for line in lines:
        parsed = _assignment(line)
        if parsed and parsed[0] == "FLASK_SECRET_KEY":
            found = True
            if parsed[1] in PLACEHOLDER_SECRETS:
                ending = "\r\n" if line.endswith("\r\n") else ("\n" if line.endswith("\n") else "")
                rewritten.append(f"FLASK_SECRET_KEY={secrets.token_urlsafe(32)}{ending}")
                changed = True
                continue
        rewritten.append(line)
    if not found:
        sep = "\r\n" if "\r\n" in text else "\n"
        prefix = "" if (not text or text.endswith(("\n", "\r"))) else sep
        rewritten.append(f"{prefix}FLASK_SECRET_KEY={secrets.token_urlsafe(32)}{sep}")
        changed = True
    return "".join(rewritten), changed


def ensure_customer_env(root: Path) -> str:
    """Return created, updated-secret, or unchanged."""
    root = root.resolve()
    env_path = root / ".env"
    example = root / ".env.example"
    created = False
    if not env_path.is_file():
        if not example.is_file():
            raise FileNotFoundError(f"Missing {example.name}")
        env_path.write_bytes(example.read_bytes())
        created = True
    original = env_path.read_text(encoding="utf-8")
    updated, changed = _replace_placeholder_secret(original)
    if changed:
        env_path.write_text(updated, encoding="utf-8", newline="")
    if created:
        return "created"
    if changed:
        return "updated-secret"
    return "unchanged"


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else default_root()
    try:
        status = ensure_customer_env(root)
    except FileNotFoundError as exc:
        sys.stderr.write(str(exc) + "\n")
        return 1
    except OSError as exc:
        sys.stderr.write(f"Could not write .env: {exc.strerror or exc}\n")
        return 1
    if status != "unchanged":
        sys.stdout.write(status + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
