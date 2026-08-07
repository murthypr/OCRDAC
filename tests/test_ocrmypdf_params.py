"""Tests for the ocrmypdf_params config entry.

Verifies that extra OCRmyPDF flags set in ocrdac.config are parsed and
inserted BEFORE the input/output positional PDF paths, that ordering of both
OCRDAC's built-in flags and user flags is preserved, that the default
(missing key) is an empty string, and that verbose mode prints the final
command and runs without OCRmyPDF printing a usage screen.
"""

import os
import sys
import shutil
import subprocess

import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import OCRDAC
from OCRDAC import load_config, ocr_pdf_dual_image, convert_pdfs

BASE_CMD = ["ocrmypdf", "--force-ocr", "--image-dpi", "300", "-l", "eng"]

OCMYPDF_AVAILABLE = any(
    os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
    for p in os.environ["PATH"].split(os.pathsep)
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_small_pdf(path):
    """Create a small single-page image-only PDF (no text layer)."""
    Image.new("RGB", (200, 200), (255, 255, 255)).save(path)


def _write_config(tmp_path, **extra):
    """Write an ini config file and return its path."""
    ini = tmp_path / "ocrdac_test.config"
    default = (
        "[OCRDAC]\n"
        "mode = prod\n"
        "directory = ./pdfs\n"
        f"output_dir = {tmp_path}\n"
        "ocr_languages = eng\n"
        "ocr_output_file = ocr_files.txt\n"
        "non_ocr_output_file = non_ocr_files.txt\n"
        "progress_interval = 25\n"
        "overwrite = false\n"
        "skip_existing_ocr = true\n"
        "preprocessing = none\n"
        "auto_preprocessing = true\n"
        "median_filter_size = 3\n"
        "threshold = 130\n"
        "ocrdac_version = v0.4\n"
    )
    ini.write_text(default)
    if extra:
        lines = ini.read_text().replace(
            "ocr_languages = eng\n",
            "ocr_languages = eng\n"
            + "\n".join(f"{k} = {v}" for k, v in extra.items())
            + "\n",
        )
        ini.write_text(lines)
    return str(ini)


@pytest.fixture
def ocr_cmd_recorder(tmp_path, monkeypatch):
    """Capture the OCRmyPDF command built by the pipeline without invoking it.

    Renders are mocked with a single white image; the OCR subprocess is faked
    to copy the (already valid) input PDF to the output path so the rest of
    the pipeline (pikepdf image replacement + metadata) can proceed.
    """
    captured = []
    orig_run = OCRDAC.subprocess.run

    monkeypatch.setattr(
        OCRDAC,
        "render_pages_to_images",
        lambda *a, **k: [Image.new("RGB", (200, 200), (255, 255, 255))],
    )

    def fake_run(cmd, **kwargs):
        if (isinstance(cmd, list) and cmd and cmd[0] == "ocrmypdf"
                and any(isinstance(c, str) and c.endswith(".pdf") for c in cmd)):
            captured.append(list(cmd))
            pdfs = [c for c in cmd if c.endswith(".pdf")]
            shutil.copy(pdfs[-2], pdfs[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    monkeypatch.setattr(OCRDAC.subprocess, "run", fake_run)
    return captured


def _convert_pdf(tmp_path, ocr_cmd_recorder, params):
    """Run the pipeline on a small PDF with the given ocrmypdf_params."""
    in_pdf = str(tmp_path / "in.pdf")
    out_pdf = str(tmp_path / "out.pdf")
    _make_small_pdf(in_pdf)
    status, msg, _ = OCRDAC.ocr_pdf_dual_image(
        in_pdf, out_pdf, "eng", ocrmypdf_params=params,
    )
    assert status in ("success", "error"), msg
    return ocr_cmd_recorder[0]


def _first_path_index(cmd):
    return next(i for i, c in enumerate(cmd) if c.endswith(".pdf"))


def _assert_flags_before_paths(cmd, flags):
    first_path = _first_path_index(cmd)
    for flag in flags:
        assert cmd.index(flag) < first_path, (
            f"{flag} must appear before input/output paths: {cmd}"
        )
    # The last two args must be the input and output PDFs.
    assert cmd[-2].endswith(".pdf"), f"Command should end with paths: {cmd}"
    assert cmd[-1].endswith(".pdf"), f"Command should end with paths: {cmd}"


# ── Test 1: Flags appear before input/output ────────────────────────────────

class TestFlagsBeforePaths:
    def test_single_verbose_before_input_pdf(self, tmp_path, ocr_cmd_recorder):
        cmd = _convert_pdf(tmp_path, ocr_cmd_recorder, "--verbose")
        _assert_flags_before_paths(cmd, ["--verbose"])
        assert cmd.index("--verbose") < _first_path_index(cmd)

    def test_builtin_flags_still_first_in_order(self, tmp_path, ocr_cmd_recorder):
        cmd = _convert_pdf(tmp_path, ocr_cmd_recorder, "--verbose")
        assert cmd[0:6] == BASE_CMD, f"Built-in flags changed: {cmd}"


# ── Test 2: Multiple flags ───────────────────────────────────────────────────

class TestMultipleFlagsBeforePaths:
    def test_flags_before_paths_in_order(self, tmp_path, ocr_cmd_recorder):
        cmd = _convert_pdf(tmp_path, ocr_cmd_recorder, "--verbose --optimize 3")
        _assert_flags_before_paths(cmd, ["--verbose", "--optimize", "3"])
        assert cmd.index("--verbose") < cmd.index("--optimize") < cmd.index("3")
        assert cmd[cmd.index("--optimize") + 1] == "3"

    def test_mixed_flag_set(self, tmp_path, ocr_cmd_recorder):
        cmd = _convert_pdf(
            tmp_path, ocr_cmd_recorder,
            "--image-dpi 300 --optimize 3 --verbose",
        )
        _assert_flags_before_paths(
            cmd, ["--image-dpi", "300", "--optimize", "3", "--verbose"],
        )


# ── Test 3: No flags (default) ───────────────────────────────────────────────

class TestNoFlags:
    def test_config_missing_key_defaults_to_empty(self, tmp_path):
        cfg = _write_config(tmp_path)
        config = load_config(cfg)
        assert config["ocrmypdf_params"] == "", (
            "Missing ocrmypdf_params should default to empty string"
        )

    def test_command_unchanged_without_params(self, tmp_path, ocr_cmd_recorder):
        cmd = _convert_pdf(tmp_path, ocr_cmd_recorder, "")
        assert cmd[0:6] == BASE_CMD, f"Built-in flags changed: {cmd}"
        assert len(cmd) == 8, f"Unexpected args appended: {cmd}"
        assert cmd[-2].endswith(".pdf") and cmd[-1].endswith(".pdf")
        assert "--verbose" not in cmd
        assert "--optimize" not in cmd


# ── Test 4: OCRmyPDF runs successfully (no usage screen) ─────────────────────

@pytest.mark.skipif(not OCMYPDF_AVAILABLE, reason="ocrmypdf not installed")
class TestRunsSuccessfully:
    def test_no_usage_screen_with_verbose(self, tmp_path, capsys):
        in_pdf = str(tmp_path / "small.pdf")
        out_pdf = str(tmp_path / "small_out.pdf")
        _make_small_pdf(in_pdf)

        status, msg, _ = OCRDAC.ocr_pdf_dual_image(
            in_pdf, out_pdf, "eng",
            ocrmypdf_params="--verbose --image-dpi 300",
        )
        captured = capsys.readouterr()

        assert status == "success", f"Pipeline failed: {msg}"
        assert "OCRmyPDF command:" in captured.out
        assert "--verbose" in captured.out
        # OCRmyPDF must not print the usage screen / abort on the flags.
        assert "usage:" not in captured.err.lower(), (
            "OCRmyPDF printed a usage screen; flags must precede paths"
        )
        assert "invalid int" not in captured.err


# ── Config loading + full pipeline threading ─────────────────────────────────

class TestConfigLoading:
    def test_reads_unquoted_value(self, tmp_path):
        cfg = _write_config(tmp_path, ocrmypdf_params="--verbose --optimize 3")
        config = load_config(cfg)
        assert config["ocrmypdf_params"] == "--verbose --optimize 3"

    def test_reads_quoted_value(self, tmp_path):
        cfg = _write_config(tmp_path, ocrmypdf_params='"--verbose"')
        config = load_config(cfg)
        assert config["ocrmypdf_params"] == "--verbose"

    def test_reads_image_dpi_value(self, tmp_path):
        cfg = _write_config(tmp_path, ocrmypdf_params="--image-dpi 300")
        config = load_config(cfg)
        assert config["ocrmypdf_params"] == "--image-dpi 300"


class TestConvertThreadsParams:
    def test_config_flow_to_pipeline(self, tmp_path, ocr_cmd_recorder):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = {
            "output_dir": str(tmp_path),
            "directory": str(tmp_path),
            "ocr_languages": "eng",
            "overwrite": "true",
            "skip_existing_ocr": "false",
            "preprocessing": "none",
            "median_filter_size": "3",
            "threshold": "130",
            "ocrdac_version": "v0.4",
            "auto_preprocessing": "true",
            "progress_interval": "25",
            "ocr_output_file": str(tmp_path / "ocr_files.txt"),
            "non_ocr_output_file": str(tmp_path / "non_ocr_files.txt"),
            "ocrmypdf_params": "--verbose --optimize 3",
        }
        convert_pdfs([in_pdf], config, str(tmp_path))
        cmd = ocr_cmd_recorder[0]
        _assert_flags_before_paths(cmd, ["--verbose", "--optimize", "3"])
        assert cmd[cmd.index("--optimize") + 1] == "3"