#!/usr/bin/env python3
"""Launch the Mun Cyber Eye Phase 5 Flask console."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.factory import create_app


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    app = create_app()
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5055"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    url = f"http://{host}:{port}"
    print(
        f"Mun Cyber Eye · {url}\n"
        "AI detects and alerts. Humans verify and decide."
    )
    if _env_flag("MUN_OPEN_BROWSER", "0"):
        def _open() -> None:
            time.sleep(1.25)
            webbrowser.open(url)

        threading.Thread(target=_open, daemon=True).start()
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
