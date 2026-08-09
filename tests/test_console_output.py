"""Tests for console output formatting — newline after progress line."""

import os
import sys
import re

import pikepdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import convert_pdfs, scan_directory, print_summary


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_blank_pdf(path):
    """Create a minimal blank PDF (no text layer — needs OCR)."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(path)
    pdf.close()


def _minimal_config(directory, tmp_path=None):
    cfg = {
        "output_dir": "",
        "directory": directory,
        "ocr_languages": "eng",
        "overwrite": "true",
        "skip_existing_ocr": "false",
        "preprocessing": "none",
        "median_filter_size": "3",
        "threshold": "130",
        "ocrdac_version": "v0.3",
        "auto_preprocessing": "false",
        "preserve_original_pdf": "false",
        "progress_interval": "25",
        "ocr_output_file": "ocr_files.txt",
        "non_ocr_output_file": "non_ocr_files.txt",
    }
    if tmp_path is not None:
        cfg["ocr_output_file"] = str(tmp_path / "ocr_files.txt")
        cfg["non_ocr_output_file"] = str(tmp_path / "non_ocr_files.txt")
    return cfg


# ── Test 1: Basic newline behavior ───────────────────────────────────────────

class TestBasicNewline:
    """Test 1 — Verify each progress line ends with \\n and next output is on a new line."""

    def test_progress_lines_end_with_newline(self, tmp_path, monkeypatch):
        pdf1 = tmp_path / "doc_a.pdf"
        pdf2 = tmp_path / "doc_b.pdf"
        _make_blank_pdf(str(pdf1))
        _make_blank_pdf(str(pdf2))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: ("success", "", {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}),
        )

        import io
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        convert_pdfs(
            [str(pdf1), str(pdf2)],
            _minimal_config(str(tmp_path)),
            str(tmp_path),
        )
        output = out.getvalue()
        lines = output.split("\n")

        idx1 = next(i for i, l in enumerate(lines) if "[1/2]" in l)
        idx2 = next(i for i, l in enumerate(lines) if "[2/2]" in l)

        assert "\u2713" in lines[idx1 + 1], (
            "Checkmark should be on the line after [1/2]"
        )
        assert "\u2713" in lines[idx2 + 1], (
            "Checkmark should be on the line after [2/2]"
        )

    def test_next_text_on_new_line(self, tmp_path, monkeypatch):
        pdf = tmp_path / "single.pdf"
        _make_blank_pdf(str(pdf))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: ("success", "", {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}),
        )

        import io
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        convert_pdfs(
            [str(pdf)],
            _minimal_config(str(tmp_path)),
            str(tmp_path),
        )
        output = out.getvalue()
        lines = output.split("\n")

        prog_line = next(l for l in lines if "[1/1]" in l)
        prog_idx = lines.index(prog_line)

        assert prog_line.endswith(" ... "), (
            "Progress line should contain '...' before newline"
        )
        assert "\u2713" in lines[prog_idx + 1], (
            "Checkmark should start on a new line after the progress line"
        )


# ── Test 2: Integration with progress dots ───────────────────────────────────

@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestIntegrationWithDots:
    """Test 2 — Real OCR pipeline: progress line ends with \\n, next output on new line."""

    def test_progress_line_newline_in_real_pipeline(self, tmp_path):
        pdf = tmp_path / "needs_ocr.pdf"
        _make_blank_pdf(str(pdf))

        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            convert_pdfs(
                [str(pdf)],
                _minimal_config(str(tmp_path)),
                str(tmp_path),
            )
            output = sys.stdout.getvalue()
        finally:
            sys.stdout = old_stdout

        lines = output.split("\n")
        prog_line = next(l for l in lines if "[1/1]" in l)
        prog_idx = lines.index(prog_line)

        assert prog_line.endswith(" ... "), (
            "Progress line should contain '...' before newline"
        )
        assert len(lines[prog_idx + 1].strip()) > 0, (
            "Next output line should not be empty"
        )
        assert not prog_line.rstrip("\n").endswith("\n"), (
            "Progress line should end with newline"
        )


# ── Test 3: No regression for other output ───────────────────────────────────

class TestNoRegression:
    """Test 3 — Full flow output: summary format, no double newlines, no broken spacing."""

    def test_output_format_unaffected(self, tmp_path, monkeypatch):
        pdf1 = tmp_path / "alpha.pdf"
        pdf2 = tmp_path / "beta.pdf"
        _make_blank_pdf(str(pdf1))
        _make_blank_pdf(str(pdf2))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: ("success", "", {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}),
        )

        import io
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        config = _minimal_config(str(tmp_path), tmp_path)

        results = scan_directory(str(tmp_path), config)
        convert_results = convert_pdfs(
            results["non_ocr"],
            config,
            str(tmp_path),
        )
        convert_pdfs(
            [str(pdf1), str(pdf2)],
            config,
            str(tmp_path),
        )

        output = out.getvalue()
        lines = output.split("\n")

        assert "=== PROD MODE:" in output
        assert any("\u2713" in l for l in lines), (
            "Checkmark symbols should be present"
        )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if "[1/2]" in stripped or "[2/2]" in stripped:
                assert stripped.endswith("..."), (
                    f"Progress line should have '...': {stripped!r}"
                )

    def test_no_double_newlines_after_progress(self, tmp_path, monkeypatch):
        pdf = tmp_path / "test.pdf"
        _make_blank_pdf(str(pdf))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: ("success", "", {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}),
        )

        import io
        out = io.StringIO()
        monkeypatch.setattr(sys, "stdout", out)

        convert_pdfs(
            [str(pdf)],
            _minimal_config(str(tmp_path)),
            str(tmp_path),
        )
        output = out.getvalue()
        lines = output.split("\n")

        prog_idx = next(i for i, l in enumerate(lines) if "[1/1]" in l)

        assert lines[prog_idx + 1].strip() == "\u2713", (
            "Line after progress should be the checkmark only (no blank line)"
        )

    def test_summary_print_unaffected(self, capsys):
        results = {
            "processed": 10,
            "ocr": [f"file{i}.pdf" for i in range(7)],
            "non_ocr": [f"file{i}.pdf" for i in range(3)],
            "errors": [],
        }
        convert_results = {"success": 2, "skipped": 1, "errors": []}
        print_summary(results, convert_results, 42)
        captured = capsys.readouterr()

        assert "OCRDAC SUMMARY" in captured.out
        assert "=" * 50 in captured.out
        assert "00:00:42" in captured.out
        assert "Successful: 2" in captured.out
        assert "Skipped:    1" in captured.out
