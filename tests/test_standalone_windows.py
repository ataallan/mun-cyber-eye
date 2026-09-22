"""Windows standalone icon, venv repair rules, and customer zip contents."""

from __future__ import annotations

import struct
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_standalone_zip  # noqa: E402
import standalone_support  # noqa: E402

ICO = ROOT / "app" / "static" / "img" / "mun-cyber-eye.ico"


def _write_python(venv: Path, import_exit: int, boot_exit: int = 0) -> None:
    exe = venv / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "code = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == '-c' else ''\n"
        "if code.strip() == 'import sys':\n"
        f"    raise SystemExit({boot_exit})\n"
        "if 'flask' in code:\n"
        f"    raise SystemExit({import_exit})\n"
        "raise SystemExit(0)\n"
    )
    exe.chmod(0o755)


def _site(venv: Path) -> Path:
    path = venv / "Lib" / "site-packages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_icon_contains_standard_png_sizes():
    data = ICO.read_bytes()
    reserved, kind, count = struct.unpack_from("<HHH", data, 0)
    assert reserved == 0
    assert kind == 1
    assert count >= 4
    widths = []
    for index in range(count):
        width, _height, _colors, _reserved, planes, bits, _size, offset = struct.unpack_from(
            "<BBBBHHII", data, 6 + index * 16
        )
        widths.append(width or 256)
        assert planes == 1
        assert bits == 32
        assert data[offset : offset + 8] == b"\x89PNG\r\n\x1a\n"
    for expected in (16, 24, 32, 48, 64, 128, 256):
        assert expected in widths


def test_optional_ml_names_cover_pip_tilde_leftovers():
    assert standalone_support.is_optional_ml_name("~orch")
    assert standalone_support.is_optional_ml_name("~orch-2.2.0.dist-info")
    assert standalone_support.is_optional_ml_name("~orchvision")
    assert standalone_support.is_optional_ml_name("~ltralytics")
    assert standalone_support.is_optional_ml_name("ultralytics-8.0.0.dist-info")
    assert not standalone_support.is_optional_ml_name("~umpy")
    assert not standalone_support.is_optional_ml_name("~lask")
    assert not standalone_support.is_optional_ml_name("flask-3.0.0.dist-info")


def test_healthy_metadata_is_not_a_corrupt_marker(tmp_path: Path):
    info = tmp_path / "flask-3.0.0.dist-info"
    info.mkdir()
    (info / "METADATA").write_text("Metadata-Version: 2.1\n", encoding="utf-8")
    assert standalone_support.distribution_marker(info) is None


def test_missing_venv_is_recreated(tmp_path: Path):
    report = standalone_support.assess_venv(tmp_path / ".venv")
    assert report["action"] == "recreate"
    assert report["reason"] == "missing"


def test_missing_core_imports_install_without_deleting(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=1)
    _site(venv)
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "install"
    assert report["reason"] == "incomplete"


def test_torch_tilde_leftover_with_healthy_imports_is_scrubbed(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=0)
    site = _site(venv)
    (site / "~orch").mkdir()
    (site / "flask").mkdir()
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "scrub_optional"
    assert "~orch" in report["markers"]

    removed = standalone_support.scrub_optional(venv)
    assert removed == ["~orch"]
    assert not (site / "~orch").exists()
    assert (site / "flask").is_dir()
    assert standalone_support.assess_venv(venv)["action"] == "ready"


def test_torch_tilde_leftover_with_failed_imports_recreates(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=1)
    site = _site(venv)
    (site / "~orch").mkdir()
    (site / "~ltralytics").mkdir()
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "recreate"
    assert report["reason"] == "damaged"
    assert "~orch" in report["markers"]


def test_core_tilde_leftover_recreates_even_when_imports_work(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=0)
    site = _site(venv)
    (site / "~lask").mkdir()
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "recreate"
    assert report["reason"] == "damaged"


def test_invalid_optional_dist_info_is_scrubbed(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=0)
    site = _site(venv)
    (site / "ultralytics-8.3.0.dist-info").mkdir()
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "scrub_optional"
    removed = standalone_support.scrub_optional(venv)
    assert removed == ["ultralytics-8.3.0.dist-info"]


def test_invalid_core_dist_info_recreates(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=0)
    site = _site(venv)
    (site / "flask-3.0.0.dist-info").mkdir()
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "recreate"


def test_broken_interpreter_recreates(tmp_path: Path):
    venv = tmp_path / ".venv"
    _write_python(venv, import_exit=0, boot_exit=1)
    report = standalone_support.assess_venv(venv)
    assert report["action"] == "recreate"
    assert report["reason"] == "damaged"


def test_onedrive_desktop_path_asks_for_local_copy():
    assert standalone_support.cloud_locked_install_path(
        r"C:\Users\a\OneDrive\Desktop\MunCyberEye"
    )
    assert standalone_support.cloud_locked_install_path(
        r"C:\Users\a\OneDrive - Mun\Desktop\Mun Cyber Eye"
    )
    assert standalone_support.cloud_locked_install_path(
        r"C:\Users\a\OneDrive\Desktop\MunCyberEye",
        desktop_path=r"C:\Users\a\OneDrive\Desktop",
    )
    assert not standalone_support.cloud_locked_install_path(r"C:\MunCyberEye")
    assert not standalone_support.cloud_locked_install_path(
        r"C:\Users\a\Desktop\MunCyberEye",
        desktop_path=r"C:\Users\a\Desktop",
    )


def test_requirements_keep_ultralytics_optional():
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        assert "ultralytics" not in stripped
        assert not stripped.startswith("torch")


def test_launchers_check_core_imports_and_skip_pip():
    statement = standalone_support.CRITICAL_IMPORT_STATEMENT
    bat = (ROOT / "Start Mun Cyber Eye.bat").read_text(encoding="utf-8")
    ps1 = (ROOT / "start_eye.ps1").read_text(encoding="utf-8")
    assert statement in bat
    assert statement in ps1
    assert "pip" not in bat.lower()
    assert "MUN_OPEN_BROWSER=1" in bat
    assert 'MUN_OPEN_BROWSER = "1"' in ps1


def test_installer_wires_shortcuts_log_and_self_heal():
    ps1 = (ROOT / "install_and_run.ps1").read_text(encoding="utf-8")
    assert "IconLocation" in ps1
    assert "WorkingDirectory" in ps1
    assert "install.log" in ps1
    assert r"C:\MunCyberEye" in ps1
    assert "Start only" in ps1
    assert "scrub-optional" in ps1
    assert "standalone_support.py" in ps1
    assert "Mun Cyber Eye.lnk" in ps1
    assert "Start Mun Cyber Eye.bat" in ps1
    wrapper = (ROOT / "install_and_run.bat").read_text(encoding="utf-8")
    assert "ExecutionPolicy Bypass" in wrapper
    assert "install_and_run.ps1" in wrapper


def test_windows_scripts_use_crlf():
    for name in (
        "install_and_run.bat",
        "install_and_run.ps1",
        "Start Mun Cyber Eye.bat",
        "start_eye.ps1",
        "scripts/repair_bundled_runtime.ps1",
        "scripts/build_windows_installer.ps1",
    ):
        data = (ROOT / name).read_bytes()
        assert b"\r\n" in data, name
        assert b"\n" not in data.replace(b"\r\n", b""), name


def test_should_skip_installer_build_output(tmp_path: Path):
    built = tmp_path / "build" / "windows-installer" / "payload" / "python.exe"
    built.parent.mkdir(parents=True)
    built.write_bytes(b"exe")
    setup = tmp_path / "dist" / "MunCyberEyeSetup.exe"
    setup.parent.mkdir()
    setup.write_bytes(b"exe")
    assert build_standalone_zip.should_skip(built, tmp_path)
    assert build_standalone_zip.should_skip(setup, tmp_path)


def test_should_skip_logs_shortcuts_and_secrets(tmp_path: Path):
    (tmp_path / "install.log").write_text("pip\n", encoding="utf-8")
    (tmp_path / "Mun Cyber Eye.lnk").write_bytes(b"lnk")
    (tmp_path / ".env").write_text("RESEND_API_KEY=secret\n", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("ok\n", encoding="utf-8")
    assert build_standalone_zip.should_skip(tmp_path / "install.log", tmp_path)
    assert build_standalone_zip.should_skip(tmp_path / "Mun Cyber Eye.lnk", tmp_path)
    assert build_standalone_zip.should_skip(tmp_path / ".env", tmp_path)
    assert not build_standalone_zip.should_skip(tmp_path / "keep.txt", tmp_path)


def test_missing_required_names_customer_files(tmp_path: Path):
    missing = build_standalone_zip.missing_required(tmp_path)
    assert "app/static/img/mun-cyber-eye.ico" in missing
    assert "Start Mun Cyber Eye.bat" in missing
    assert "start_eye.ps1" in missing


def test_customer_zip_contains_icon_and_launchers(tmp_path: Path):
    dest = tmp_path / "mun-cyber-eye-standalone.zip"
    count = build_standalone_zip.build_zip(ROOT, dest)
    assert count > 0
    with zipfile.ZipFile(dest) as archive:
        names = set(archive.namelist())
    for rel in build_standalone_zip.REQUIRED_ZIP_PATHS:
        assert rel in names
    assert ".env" not in names
    assert "install.log" not in names
    assert not any(name.endswith(".lnk") for name in names)
    assert not any(name.startswith(".venv/") or name.startswith(".git/") for name in names)


def test_build_zip_refuses_incomplete_tree(tmp_path: Path):
    with pytest.raises(SystemExit):
        build_standalone_zip.build_zip(tmp_path, tmp_path / "out.zip")
