#!/usr/bin/env python3
"""Check the Windows Setup project, and build MunCyberEyeSetup.exe on Windows.

Customers install MunCyberEyeSetup.exe. This script does not freeze OpenCV
into a one-file executable. On Windows it runs scripts/build_windows_installer.ps1,
which bundles embeddable Python, installs requirements.txt, and compiles the
Inno Setup script.

From the repo root on Windows:

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\build_windows_installer.ps1

On another operating system this command checks the installer project and
prints that same build command. Output of a Windows build:

    dist/MunCyberEyeSetup.exe
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNTIME_JSON = ROOT / "installer" / "runtime.json"
VERSION_FILE = ROOT / "installer" / "VERSION"
ISS_FILE = ROOT / "installer" / "MunCyberEye.iss"

INCLUDE_DIRS = ("app", "alerts", "ingest", "risk", "vision", "data", "sample_data")
INCLUDE_FILES = (
    "run.py",
    "pipeline.py",
    "requirements.txt",
    ".env.example",
    "Start Mun Cyber Eye.bat",
    "start_eye.ps1",
    "scripts/standalone_support.py",
    "scripts/ensure_customer_env.py",
    "scripts/repair_bundled_runtime.ps1",
)
REQUIRED_PAYLOAD = INCLUDE_FILES + (
    "app/static/img/mun-cyber-eye.ico",
    "data/checkpoints/activity_demo.joblib",
)
SKIP_DIR_NAMES = {
    ".cursor",
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "htmlcov",
    "venv",
}
SKIP_REL_PREFIXES = (
    "data/activity/test/",
    "data/activity/train/",
    "data/activity/val/",
    "data/dangerous/test/",
    "data/dangerous/train/",
    "data/dangerous/val/",
    "data/objects/test/",
    "data/objects/train/",
    "data/objects/val/",
    "data/snapshots/",
    "data/uploads/",
)
SKIP_SUFFIXES = {".avi", ".db", ".lnk", ".mkv", ".mov", ".mp4", ".onnx", ".pt", ".pyc", ".pyo"}
TEXT_SUFFIXES = {".bat", ".example", ".iss", ".json", ".md", ".ps1", ".py", ".txt", ".yml", ".yaml"}
FORBIDDEN_SNIPPETS = (
    "CF_API_TOKEN",
    "CLOUDFLARE_API_TOKEN",
    "CLOUDFLARE_TUNNEL",
    "RESEND_API_KEY=re_",
    "TUNNEL_TOKEN",
)
EMPTY_CUSTOMER_KEYS = (
    "ADMIN_PASSWORD",
    "ADMIN_USERNAME",
    "ALERT_EMAIL_RECIPIENTS",
    "ALERT_WEBHOOK_SECRET",
    "ALERT_WEBHOOK_URL",
    "DEVELOPER_PASSWORD",
    "DEVELOPER_USERNAME",
    "OPERATOR_PASSWORD",
    "RESEND_API_KEY",
    "RTSP_DEMO_URI",
    "SECURITY_ALERT_EMAIL",
)
SECRET_PLACEHOLDER = "change-me-to-a-long-random-string"
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+$")
ISS_MARKERS = (
    "AppId={{8F4E2C1A-6B7D-4E90-9C3A-1D5F0A2B7E64}",
    "DefaultDirName={localappdata}\\MunCyberEye",
    "{userdesktop}\\Mun Cyber Eye",
    'IconFilename: "{app}\\app\\static\\img\\mun-cyber-eye.ico"',
    "mun-cyber-eye.ico",
    "{uninstallexe}",
    "PrivilegesRequired=lowest",
    "Start Mun Cyber Eye.bat",
    "ensure_customer_env.py",
    "OutputBaseFilename=MunCyberEyeSetup",
    "shellexec",
)


def load_runtime() -> dict[str, str]:
    data = json.loads(RUNTIME_JSON.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("installer/runtime.json must be an object")
    return {str(key): str(value) for key, value in data.items()}


def app_version() -> str:
    return VERSION_FILE.read_text(encoding="utf-8").strip()


def embeddable_pth_text(stdlib_zip_name: str) -> str:
    """python*._pth contents that turn on site-packages for the embeddable runtime."""
    name = Path(stdlib_zip_name).name
    if not name.startswith("python") or not name.endswith(".zip"):
        raise ValueError(f"Expected a python stdlib zip name, got {stdlib_zip_name!r}")
    return f"{name}\r\n.\r\n\r\nLib\\site-packages\r\nimport site\r\n"


def write_embeddable_pth(stdlib_zip_name: str, dest: Path) -> None:
    dest.write_bytes(embeddable_pth_text(stdlib_zip_name).encode("ascii"))


def _env_assignments(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for line in text.splitlines():
        body = line.strip()
        if not body or body.startswith("#") or "=" not in body:
            continue
        key, value = body.split("=", 1)
        key = key.strip()
        if not key or any(ch.isspace() for ch in key):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        found[key] = value
    return found


def customer_env_problems(example_text: str) -> list[str]:
    """Reject a shipped .env.example that already contains real secrets."""
    values = _env_assignments(example_text)
    problems: list[str] = []
    for key in EMPTY_CUSTOMER_KEYS:
        if values.get(key, ""):
            problems.append(f".env.example sets {key}")
    secret = values.get("FLASK_SECRET_KEY", "")
    if secret != SECRET_PLACEHOLDER:
        problems.append("FLASK_SECRET_KEY in .env.example must stay the documented placeholder")
    return problems


def _skip_payload_file(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in SKIP_DIR_NAMES for part in rel.parts):
        return True
    rel_posix = rel.as_posix()
    if rel_posix.startswith(SKIP_REL_PREFIXES) and path.name not in {".gitkeep", "README.md"}:
        return True
    if path.name == ".env" or path.suffix.lower() in SKIP_SUFFIXES:
        return True
    return False


def payload_relative_paths(root: Path) -> list[str]:
    rels: list[str] = []
    for rel in INCLUDE_FILES:
        path = root / rel
        if path.is_file() and not _skip_payload_file(path, root):
            rels.append(rel)
    for dirname in INCLUDE_DIRS:
        base = root / dirname
        if not base.is_dir():
            continue
        for path in base.rglob("*"):
            if path.is_file() and not _skip_payload_file(path, root):
                rels.append(path.relative_to(root).as_posix())
    return sorted(set(rels))


def forbidden_snippets(path: Path) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".env.example":
        return []
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeError:
        return []
    return [snippet for snippet in FORBIDDEN_SNIPPETS if snippet in text]


def payload_content_problems(root: Path, rels: list[str] | None = None) -> list[str]:
    problems: list[str] = []
    chosen = payload_relative_paths(root) if rels is None else rels
    for rel in chosen:
        if Path(rel).name == ".env":
            problems.append(f"payload includes a real .env: {rel}")
        for snippet in forbidden_snippets(root / rel):
            problems.append(f"{rel} contains {snippet}")
    return problems


def validate_project(root: Path = ROOT) -> list[str]:
    problems: list[str] = []
    required = (
        RUNTIME_JSON,
        VERSION_FILE,
        ISS_FILE,
        root / "installer" / "INSTALL.txt",
        root / "scripts" / "build_windows_installer.ps1",
        root / "scripts" / "repair_bundled_runtime.ps1",
        root / "scripts" / "ensure_customer_env.py",
        root / "scripts" / "stage_windows_payload.py",
        root / ".github" / "workflows" / "windows-installer.yml",
        root / "docs" / "WINDOWS_INSTALLER.md",
    )
    for path in required:
        if not path.is_file():
            problems.append(f"missing {path.relative_to(root).as_posix()}")
    if problems:
        return problems
    try:
        runtime = json.loads((root / "installer" / "runtime.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"installer/runtime.json is invalid: {exc}"]
    for key in ("python_embed_url", "get_pip_url", "inno_url", "output_exe", "install_subdir"):
        if key not in runtime:
            problems.append(f"installer/runtime.json missing {key}")
    embed_url = str(runtime.get("python_embed_url", ""))
    inno_url = str(runtime.get("inno_url", ""))
    if not embed_url.startswith("https://www.python.org/ftp/python/"):
        problems.append("python_embed_url must be an https://www.python.org/ftp/python/ URL")
    if "embed-amd64.zip" not in embed_url:
        problems.append("python_embed_url must name the Windows amd64 embeddable zip")
    if not inno_url.startswith("https://github.com/jrsoftware/issrc/releases/download/"):
        problems.append("inno_url must be the pinned jrsoftware Inno Setup download")
    version = (root / "installer" / "VERSION").read_text(encoding="utf-8").strip()
    if not VERSION_RE.match(version):
        problems.append("installer/VERSION must be major.minor.patch")
    iss = (root / "installer" / "MunCyberEye.iss").read_text(encoding="utf-8")
    for marker in ISS_MARKERS:
        if marker not in iss:
            problems.append(f"installer script is missing {marker}")
    if iss.count("{userdesktop}") != 1:
        problems.append("installer must create exactly one Desktop shortcut")
    example = root / ".env.example"
    if example.is_file():
        problems.extend(customer_env_problems(example.read_text(encoding="utf-8")))
    rels = payload_relative_paths(root)
    for rel in REQUIRED_PAYLOAD:
        if rel not in rels:
            problems.append(f"installer payload is missing {rel}")
    for banned in ("install_and_run.ps1", "install_and_run.bat", ".env"):
        if banned in rels:
            problems.append(f"installer payload must not ship {banned}")
    problems.extend(payload_content_problems(root, rels))
    bat = (root / "Start Mun Cyber Eye.bat").read_text(encoding="utf-8")
    ps1 = (root / "start_eye.ps1").read_text(encoding="utf-8")
    if r"python\python.exe" not in bat or r"python\python.exe" not in ps1:
        problems.append("launchers must prefer the bundled python\\python.exe runtime")
    if bat.find(r"python\python.exe") > bat.find(r".venv\Scripts\python.exe"):
        problems.append("the Desktop launcher must select bundled Python before .venv")
    build_ps1 = (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8")
    for marker in ("ISCC.exe", "runtime.json", "get-pip.py", "stage_windows_payload.py", "--write-pth"):
        if marker not in build_ps1:
            problems.append(f"build_windows_installer.ps1 is missing {marker}")
    repair = (root / "scripts" / "repair_bundled_runtime.ps1").read_text(encoding="utf-8")
    if "--no-index" not in repair or "requirements.txt" not in repair:
        problems.append("repair script must reinstall requirements from the offline wheels")
    workflow = (root / ".github" / "workflows" / "windows-installer.yml").read_text(encoding="utf-8")
    if "windows-latest" not in workflow or "build_windows_installer.ps1" not in workflow:
        problems.append("GitHub Actions must build the Setup exe on windows-latest")
    if "MunCyberEyeSetup.exe" not in workflow:
        problems.append("GitHub Actions must upload MunCyberEyeSetup.exe")
    return problems


def stage_payload(root: Path, dest: Path) -> list[str]:
    import shutil

    root = root.resolve()
    dest = dest.resolve()
    problems = customer_env_problems((root / ".env.example").read_text(encoding="utf-8"))
    rels = payload_relative_paths(root)
    for rel in REQUIRED_PAYLOAD:
        if rel not in rels:
            problems.append(f"missing {rel}")
    problems.extend(payload_content_problems(root, rels))
    if problems:
        raise SystemExit("Refusing to stage the Windows payload: " + "; ".join(problems))
    if dest.exists():
        shutil.rmtree(dest)
    for rel in rels:
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    return rels


def windows_build_instructions() -> str:
    return "\n".join(
        (
            "Build MunCyberEyeSetup.exe on Windows from the repo root:",
            "",
            "    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\\build_windows_installer.ps1",
            "",
            "The script downloads embeddable Python, installs requirements.txt into that runtime,",
            "vendors wheels for offline repair, and compiles installer/MunCyberEye.iss.",
            "Output: dist\\MunCyberEyeSetup.exe",
            "",
            "GitHub Actions workflow \"Windows installer\" runs the same script on windows-latest",
            "and uploads the MunCyberEyeSetup artifact.",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check or build the Mun Cyber Eye Windows Setup")
    parser.add_argument("--check", action="store_true", help="Validate the installer project and exit")
    parser.add_argument(
        "--write-pth",
        nargs=2,
        metavar=("STDLIB_ZIP", "DEST"),
        help="Write an embeddable python*._pth file that enables site-packages",
    )
    args = parser.parse_args(argv)
    if args.write_pth:
        stdlib_zip, dest = args.write_pth
        write_embeddable_pth(stdlib_zip, Path(dest))
        return 0

    problems = validate_project(ROOT)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    if args.check or sys.platform != "win32":
        if not args.check:
            print(windows_build_instructions())
            print("")
            print("This machine is not Windows, so MunCyberEyeSetup.exe was not compiled.")
            print("The installer project checked successfully.")
        return 0

    script = ROOT / "scripts" / "build_windows_installer.ps1"
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(script),
        ],
        cwd=ROOT,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
