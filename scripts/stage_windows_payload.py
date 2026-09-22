#!/usr/bin/env python3
"""Copy the customer app tree into the Inno Setup payload directory.

The Windows build script calls this before it adds embeddable Python.
A real .env, git metadata, tests, and the source-folder installer are omitted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import build_windows_installer  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage the Mun Cyber Eye installer payload")
    parser.add_argument("--dest", required=True, help="Payload directory to create")
    parser.add_argument("--root", default=str(ROOT), help="Repository root")
    args = parser.parse_args(argv)
    rels = build_windows_installer.stage_payload(Path(args.root), Path(args.dest))
    print(f"Staged {len(rels)} files into {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
