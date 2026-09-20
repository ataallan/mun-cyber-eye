#!/usr/bin/env python3
"""Launch the Mun Cyber Eye Phase 4 Flask console."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from app.factory import create_app


def main() -> None:
    app = create_app()
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5055"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    print(
        f"Mun Cyber Eye Phase 4 · http://{host}:{port}\n"
        "AI detects and alerts. Humans verify and decide.\n"
        "Authorized cameras only — no autonomous enforcement."
    )
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
