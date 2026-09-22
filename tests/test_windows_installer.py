"""Windows Setup project: payload rules, secret handling, and launcher wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_windows_installer  # noqa: E402
import ensure_customer_env  # noqa: E402


def test_project_validates():
    assert build_windows_installer.validate_project(ROOT) == []


def test_iss_creates_one_desktop_shortcut():
    iss = (ROOT / "installer" / "MunCyberEye.iss").read_text(encoding="utf-8")
    assert iss.count("{userdesktop}") == 1
    assert r"{localappdata}\MunCyberEye" in iss
    assert "mun-cyber-eye.ico" in iss
    assert "{uninstallexe}" in iss
    assert "PrivilegesRequired=lowest" in iss
    runtime = json.loads((ROOT / "installer" / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["install_subdir"] == "MunCyberEye"
    assert runtime["shortcut_name"] == "Mun Cyber Eye"
    assert runtime["output_exe"] == "MunCyberEyeSetup.exe"


def test_embeddable_pth_enables_site_packages():
    text = build_windows_installer.embeddable_pth_text("python312.zip")
    assert "import site" in text
    assert "# import site" not in text
    assert r"Lib\site-packages" in text
    assert text.startswith("python312.zip\r\n")
    with pytest.raises(ValueError):
        build_windows_installer.embeddable_pth_text("not-python.txt")


def test_env_example_has_no_customer_secrets():
    text = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert build_windows_installer.customer_env_problems(text) == []
    assert build_windows_installer.customer_env_problems("RESEND_API_KEY=re_live_secret\n")


def test_payload_lists_icon_and_omits_secrets_and_zip_installer():
    rels = build_windows_installer.payload_relative_paths(ROOT)
    assert "app/static/img/mun-cyber-eye.ico" in rels
    assert "data/checkpoints/activity_demo.joblib" in rels
    assert "scripts/ensure_customer_env.py" in rels
    assert ".env" not in rels
    assert "install_and_run.ps1" not in rels
    assert "install_and_run.bat" not in rels
    assert not any(rel.startswith("tests/") or rel.startswith(".git/") for rel in rels)
    assert build_windows_installer.payload_content_problems(ROOT, rels) == []


def test_stage_payload_copies_app_and_refuses_env(tmp_path: Path):
    dest = tmp_path / "payload"
    rels = build_windows_installer.stage_payload(ROOT, dest)
    assert (dest / "app" / "static" / "img" / "mun-cyber-eye.ico").is_file()
    assert (dest / "Start Mun Cyber Eye.bat").is_file()
    assert (dest / ".env.example").is_file()
    assert not (dest / ".env").exists()
    assert len(rels) == len(list(dest.rglob("*"))) - len([p for p in dest.rglob("*") if p.is_dir()])
    assert all((dest / rel).is_file() for rel in rels)


def test_launchers_prefer_bundled_runtime_and_keep_browser_flag():
    bat = (ROOT / "Start Mun Cyber Eye.bat").read_text(encoding="utf-8")
    ps1 = (ROOT / "start_eye.ps1").read_text(encoding="utf-8")
    assert bat.find(r"python\python.exe") < bat.find(r".venv\Scripts\python.exe")
    assert r"python\python.exe" in ps1
    assert "ensure_customer_env.py" in bat
    assert "ensure_customer_env.py" in ps1
    assert "repair_bundled_runtime.ps1" in bat
    assert "MUN_OPEN_BROWSER=1" in bat
    assert 'MUN_OPEN_BROWSER = "1"' in ps1
    assert "pip" not in bat.lower()


def test_repair_script_uses_offline_wheels_and_crlf():
    repair = ROOT / "scripts" / "repair_bundled_runtime.ps1"
    text = repair.read_text(encoding="utf-8")
    assert "--no-index" in text
    assert "requirements.txt" in text
    assert "wheels" in text
    data = repair.read_bytes()
    assert b"\r\n" in data
    assert b"\n" not in data.replace(b"\r\n", b"")
    build = (ROOT / "scripts" / "build_windows_installer.ps1").read_bytes()
    assert b"\r\n" in build
    assert b"\n" not in build.replace(b"\r\n", b"")


def test_ensure_customer_env_generates_secret_once(tmp_path: Path):
    example = ROOT / ".env.example"
    (tmp_path / ".env.example").write_bytes(example.read_bytes())
    status = ensure_customer_env.ensure_customer_env(tmp_path)
    assert status == "created"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "change-me-to-a-long-random-string" not in env_text
    assert "RESEND_API_KEY=\n" in env_text.replace("\r\n", "\n") or "RESEND_API_KEY=\r\n" in env_text
    assert "ADMIN_PASSWORD=\n" in env_text.replace("\r\n", "\n") or env_text.replace("\r\n", "\n").count("ADMIN_PASSWORD=\n")
    secret_lines = [line for line in env_text.splitlines() if line.startswith("FLASK_SECRET_KEY=")]
    assert len(secret_lines) == 1
    secret = secret_lines[0].split("=", 1)[1]
    assert secret not in ensure_customer_env.PLACEHOLDER_SECRETS
    assert len(secret) >= 32
    again = ensure_customer_env.ensure_customer_env(tmp_path)
    assert again == "unchanged"
    assert (tmp_path / ".env").read_text(encoding="utf-8") == env_text


def test_ensure_customer_env_keeps_a_real_secret(tmp_path: Path):
    (tmp_path / ".env").write_text(
        "FLASK_SECRET_KEY=already-set-by-the-customer-123456\nRESEND_API_KEY=\n",
        encoding="utf-8",
    )
    assert ensure_customer_env.ensure_customer_env(tmp_path) == "unchanged"
    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "FLASK_SECRET_KEY=already-set-by-the-customer-123456" in text
    assert "RESEND_API_KEY=\n" in text


def test_build_instructions_name_the_setup_command():
    text = build_windows_installer.windows_build_instructions()
    assert "build_windows_installer.ps1" in text
    assert "MunCyberEyeSetup.exe" in text
