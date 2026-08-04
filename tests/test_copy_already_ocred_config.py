"""Tests for the copy_already_ocred_files configuration flag.

Covers:
1. Flag = true  -> already-OCR'd PDFs are copied, listed, counted.
2. Flag = false -> already-OCR'd PDFs are skipped entirely.
3. Default (parameter missing) -> behaves as true.
4. Mixed directory (already-OCR'd + non-OCR PDFs) in both modes.
"""

import os
import sys
import filecmp

import pikepdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import (
    load_config,
    scan_directory,
    copy_ocr_pdfs,
    convert_pdfs,
    pdf_needs_ocr,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_pdf_with_text(path, text="Hello World"):
    """Create a minimal PDF with an existing text layer (already OCR'd)."""
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page(page_size=(612, 792))
    content_data = f"BT /F1 12 Tf 100 700 Td ({text}) Tj ET".encode()
    content_stream = pikepdf.Stream(pdf, content_data)
    page.Contents = content_stream
    page.Resources = pikepdf.Dictionary({
        "/Font": pikepdf.Dictionary({
            "/F1": pikepdf.Dictionary({
                "/Type": pikepdf.Name("/Font"),
                "/Subtype": pikepdf.Name("/Type1"),
                "/BaseFont": pikepdf.Name("/Helvetica"),
            })
        })
    })
    pdf.save(path)
    pdf.close()


def _make_blank_pdf(path):
    """Create a minimal blank PDF (no text layer — needs OCR)."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(path)
    pdf.close()


def _config_for(directory, output_dir, tmp_path, flag):
    cfg = {
        "output_dir": output_dir or "",
        "directory": directory,
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
        "copy_already_ocred_files": str(flag).lower(),
    }
    return cfg


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _run_prod(src_dir, out_dir, tmp_path, flag):
    """Replicate the prod-mode main loop: scan -> copy -> convert."""
    config = _config_for(str(src_dir), str(out_dir), tmp_path, flag)
    results = scan_directory(str(src_dir), config)
    ocr_copied = copy_ocr_pdfs(results["ocr"], config, str(src_dir))
    convert_results = None
    if results["non_ocr"]:
        convert_results = convert_pdfs(results["non_ocr"], config, str(src_dir))
    return results, config, ocr_copied, convert_results


# ── Test 1: Flag = true (copy enabled) ───────────────────────────────────────

class TestFlagTrue:
    """Test 1 — copy_already_ocred_files = true copies already-OCR'd PDFs."""

    def test_file_copied_to_output_directory(self, tmp_path):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)
        assert not pdf_needs_ocr(pdf_path), "PDF should already have text"

        results, config, ocr_copied, _ = _run_prod(
            src_dir, out_dir, tmp_path, flag=True
        )

        assert ocr_copied == 1
        dest = out_dir / "has_text.pdf"
        assert dest.exists(), "Copied file should exist in output directory"
        assert filecmp.cmp(pdf_path, str(dest), shallow=False), (
            "Copied file should be byte-identical to source"
        )

    def test_file_appears_in_ocr_files_txt(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)

        results, config, _, _ = _run_prod(src_dir, tmp_path / "out", tmp_path, flag=True)

        contents = _read(config["ocr_output_file"])
        assert pdf_path in contents, "ocr_files.txt should contain the path"

    def test_ocr_count_increments(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        _make_pdf_with_text(str(src_dir / "a.pdf"))
        _make_pdf_with_text(str(src_dir / "b.pdf"))

        results, config, ocr_copied, _ = _run_prod(
            src_dir, tmp_path / "out", tmp_path, flag=True
        )

        assert results["ocr_count"] == 2
        assert ocr_copied == 2

    def test_no_ocr_performed(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        _make_pdf_with_text(str(src_dir / "has_text.pdf"))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: pytest.fail("OCR must not run on already-OCR'd PDFs"),
        )

        results, config, ocr_copied, convert_results = _run_prod(
            src_dir, out_dir, tmp_path, flag=True
        )

        assert len(results["non_ocr"]) == 0, "No files should need OCR"
        assert convert_results is None, "No conversion step should run"
        assert ocr_copied == 1
        assert (out_dir / "has_text.pdf").exists()


# ── Test 2: Flag = false (copy disabled) ─────────────────────────────────────

class TestFlagFalse:
    """Test 2 — copy_already_ocred_files = false skips already-OCR'd PDFs."""

    def test_file_not_copied(self, tmp_path):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)

        results, config, ocr_copied, _ = _run_prod(
            src_dir, out_dir, tmp_path, flag=False
        )

        assert ocr_copied == 0
        assert not (out_dir / "has_text.pdf").exists(), (
            "Skipped file must NOT be copied to output directory"
        )

    def test_file_not_in_ocr_files_txt(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)

        results, config, _, _ = _run_prod(src_dir, tmp_path / "out", tmp_path, flag=False)

        contents = _read(config["ocr_output_file"])
        assert pdf_path not in contents, (
            "ocr_files.txt must NOT contain the skipped path"
        )
        assert contents.strip() == "", "ocr_files.txt should be empty"

    def test_ocr_count_not_incremented(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        _make_pdf_with_text(str(src_dir / "has_text.pdf"))

        results, config, ocr_copied, _ = _run_prod(
            src_dir, tmp_path / "out", tmp_path, flag=False
        )

        assert results["ocr_count"] == 0
        assert len(results["ocr"]) == 0
        assert ocr_copied == 0

    def test_no_ocr_performed(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        _make_pdf_with_text(str(src_dir / "has_text.pdf"))

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: pytest.fail("OCR must not run on skipped PDFs"),
        )

        results, config, ocr_copied, convert_results = _run_prod(
            src_dir, tmp_path / "out", tmp_path, flag=False
        )

        assert len(results["non_ocr"]) == 0
        assert convert_results is None
        assert ocr_copied == 0


# ── Test 3: Default behavior (parameter missing) ─────────────────────────────

class TestDefaultBehavior:
    """Test 3 — missing parameter defaults to true (copy enabled)."""

    def _write_config(self, tmp_path, src_dir, out_dir):
        ini = tmp_path / "ocrdac_test.config"
        ini.write_text(
            f"[OCRDAC]\n"
            f"mode = prod\n"
            f"directory = {src_dir}\n"
            f"output_dir = {out_dir}\n"
            f"ocr_languages = eng\n"
            f"ocr_output_file = {tmp_path / 'ocr_files.txt'}\n"
            f"non_ocr_output_file = {tmp_path / 'non_ocr_files.txt'}\n"
            f"progress_interval = 25\n"
            f"overwrite = false\n"
            f"skip_existing_ocr = true\n"
            f"preprocessing = none\n"
            f"auto_preprocessing = true\n"
            f"median_filter_size = 3\n"
            f"threshold = 130\n"
            f"ocrdac_version = v0.4\n"
        )
        return str(ini)

    def test_config_defaults_to_true(self, tmp_path):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        ini = self._write_config(tmp_path, src_dir, out_dir)
        config = load_config(ini)

        assert config["copy_already_ocred_files"] == "true", (
            "Missing parameter should default to true"
        )

    def test_default_runs_as_copy_enabled(self, tmp_path):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)

        ini = self._write_config(tmp_path, src_dir, out_dir)
        config = load_config(ini)
        config["directory"] = str(src_dir)

        results = scan_directory(str(src_dir), config)
        ocr_copied = copy_ocr_pdfs(results["ocr"], config, str(src_dir))

        assert ocr_copied == 1
        assert (out_dir / "has_text.pdf").exists()
        assert pdf_path in _read(config["ocr_output_file"])


# ── Test 4: Mixed directory ──────────────────────────────────────────────────

class TestMixedDirectory:
    """Test 4 — already-OCR'd + non-OCR PDFs in the same directory."""

    def _make_mixed(self, src_dir):
        ocrd_pdf = str(src_dir / "already_ocrd.pdf")
        blank_pdf = str(src_dir / "needs_ocr.pdf")
        _make_pdf_with_text(ocrd_pdf)
        _make_blank_pdf(blank_pdf)
        return ocrd_pdf, blank_pdf

    def test_flag_true(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        ocrd_pdf, blank_pdf = self._make_mixed(src_dir)

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: (
                "success", "",
                {"clean_pages": 1, "preprocessed_pages": 0, "preprocessing_reasons": {}},
            ),
        )

        results, config, ocr_copied, convert_results = _run_prod(
            src_dir, out_dir, tmp_path, flag=True
        )

        assert results["processed"] == 2
        assert results["ocr_count"] == 1
        assert ocr_copied == 1
        assert (out_dir / "already_ocrd.pdf").exists()
        assert filecmp.cmp(ocrd_pdf, str(out_dir / "already_ocrd.pdf"), shallow=False)
        assert ocrd_pdf in _read(config["ocr_output_file"])

        assert len(results["non_ocr"]) == 1
        assert convert_results["success"] == 1, (
            "Non-OCR file should still be converted"
        )

    def test_flag_false(self, tmp_path, monkeypatch):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        ocrd_pdf, blank_pdf = self._make_mixed(src_dir)

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: (
                "success", "",
                {"clean_pages": 1, "preprocessed_pages": 0, "preprocessing_reasons": {}},
            ),
        )

        results, config, ocr_copied, convert_results = _run_prod(
            src_dir, out_dir, tmp_path, flag=False
        )

        assert results["processed"] == 2
        assert results["ocr_count"] == 0
        assert ocr_copied == 0
        assert not (out_dir / "already_ocrd.pdf").exists(), (
            "Already-OCR'd file must NOT be copied when flag is false"
        )
        assert ocrd_pdf not in _read(config["ocr_output_file"])

        assert len(results["non_ocr"]) == 1
        assert convert_results["success"] == 1, (
            "Non-OCR file should still be converted"
        )
