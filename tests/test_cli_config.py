"""Tests for the --config CLI flag (custom config file selection)."""

import os
import subprocess
import sys

import pytest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(BASE_DIR, "OCRDAC.py")

DEFAULT_OUT_FILES = (
    os.path.join(BASE_DIR, "ocr_files.txt"),
    os.path.join(BASE_DIR, "non_ocr_files.txt"),
)


def _run_cli(*args, timeout=120):
    return subprocess.run(
        [sys.executable, SCRIPT, *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=BASE_DIR,
    )


def _cleanup_default_output_files():
    for path in DEFAULT_OUT_FILES:
        try:
            os.remove(path)
        except OSError:
            pass


def _write_custom_config(tmp_path, src_dir=None, mode="dry-run",
                         ocr_file="ocr_custom.txt", non_ocr_file="non_ocr_custom.txt"):
    cfg_path = tmp_path / "custom.config"
    if src_dir is None:
        src_dir = tmp_path / "src"
        src_dir.mkdir(exist_ok=True)
    cfg_path.write_text(
        f"[OCRDAC]\n"
        f"mode = {mode}\n"
        f"directory = {src_dir}\n"
        f"ocr_languages = eng\n"
        f"progress_interval = 25\n"
        f"ocr_output_file = {tmp_path / ocr_file}\n"
        f"non_ocr_output_file = {tmp_path / non_ocr_file}\n"
    )
    return str(cfg_path)


class TestConfigFlag:
    """Test 1 — --config <path> uses the given file instead of ocrdac.config."""

    def test_config_flag_selects_custom_file(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        cfg_path = _write_custom_config(tmp_path, src_dir=str(src))
        ocr_file = tmp_path / "ocr_custom.txt"
        non_ocr_file = tmp_path / "non_ocr_custom.txt"

        result = _run_cli("--config", cfg_path)

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Mode: DRY-RUN" in result.stdout, (
            "Custom config mode (dry-run) should be applied"
        )
        assert f"Scanning: {src}" in result.stdout, (
            "Custom config directory should be used (not the default)"
        )
        assert ocr_file.exists(), "Custom ocr_output_file should be written"
        assert non_ocr_file.exists(), "Custom non_ocr_output_file should be written"
        _cleanup_default_output_files()

    def test_config_flag_equals_form(self, tmp_path):
        src = tmp_path / "src2"
        src.mkdir()
        cfg_path = _write_custom_config(tmp_path, src_dir=str(src))

        result = _run_cli(f"--config={cfg_path}")

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert f"Scanning: {src}" in result.stdout
        _cleanup_default_output_files()

    def test_config_flag_positional_args_unchanged(self, tmp_path):
        src = tmp_path / "src3"
        src.mkdir()
        cfg_path = _write_custom_config(tmp_path, src_dir=str(tmp_path / "unused"),
                                        mode="prod")

        result = _run_cli("--config", cfg_path, str(src), "dry-run")

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert f"Scanning: {src}" in result.stdout, (
            "Positional directory should still override config"
        )
        assert "Mode: DRY-RUN" in result.stdout, (
            "Positional mode should still override config"
        )
        _cleanup_default_output_files()

    def test_config_flag_missing_value(self, tmp_path):
        result = _run_cli("--config")

        assert result.returncode != 0
        assert "Error: --config requires a file path" in (
            result.stdout + result.stderr
        )

    def test_config_flag_empty_value(self, tmp_path):
        result = _run_cli("--config=")

        assert result.returncode != 0
        assert "Error: --config requires a file path" in (
            result.stdout + result.stderr
        )

    def test_config_flag_nonexistent_file(self, tmp_path):
        src = tmp_path / "src4"
        src.mkdir()
        missing = tmp_path / "missing.config"

        result = _run_cli("--config", str(missing), str(src), "dry-run")

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert f"Config file not found: {missing}" in result.stdout, (
            "Missing custom config should fall back to defaults"
        )
        assert f"Scanning: {src}" in result.stdout
        _cleanup_default_output_files()


class TestDefaultBehavior:
    """Test 2 — Without --config the default ocrdac.config next to the script is used."""

    def test_no_flag_uses_default_config(self, tmp_path):
        script_config = os.path.join(BASE_DIR, "ocrdac.config")
        with open(script_config) as fh:
            content = fh.read()
        assert "mode = prod" in content, (
            "Sanity: repo ocrdac.config should be in prod mode"
        )

        src = tmp_path / "src5"
        src.mkdir()

        result = _run_cli(str(src), "dry-run")

        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "Mode: DRY-RUN" in result.stdout, (
            "Default config loaded; positional dry-run still applies"
        )
        assert f"Scanning: {src}" in result.stdout
        _cleanup_default_output_files()