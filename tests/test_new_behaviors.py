"""Tests for new behaviors: already-OCR'd copy and auto-preprocessing stats."""

import os
import sys
import subprocess
import filecmp

import pikepdf
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import (
    ocr_pdf_dual_image,
    convert_pdfs,
    scan_directory,
    copy_ocr_pdfs,
    print_summary,
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
    """Create a minimal blank PDF (no text — needs OCR)."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(path)
    pdf.close()


def _config_for(directory, output_dir=None, tmp_path=None):
    cfg = {
        "output_dir": output_dir or "",
        "directory": directory,
        "ocr_languages": "eng",
        "overwrite": "true",
        "skip_existing_ocr": "false",
        "preprocessing": "none",
        "median_filter_size": "3",
        "threshold": "130",
        "ocrdac_version": "v0.3",
        "auto_preprocessing": "true",
        "progress_interval": "25",
        "ocr_output_file": "ocr_files.txt",
        "non_ocr_output_file": "non_ocr_files.txt",
    }
    if tmp_path is not None:
        cfg["ocr_output_file"] = str(tmp_path / "ocr_files.txt")
        cfg["non_ocr_output_file"] = str(tmp_path / "non_ocr_files.txt")
    return cfg


def _check_text_layer(pdf_path):
    """Check if a PDF has a text layer by searching for BT/ET operators."""
    with pikepdf.Pdf.open(pdf_path) as pdf:
        for page in pdf.pages:
            contents = page.get("/Contents")
            if contents is None:
                continue
            streams = contents if isinstance(contents, pikepdf.Array) else [contents]
            for stream in streams:
                try:
                    data = stream.read_bytes()
                    if b"BT" in data and b"ET" in data:
                        return True
                except Exception:
                    continue
    return False


def _create_img_pdf(path, img):
    """Create a single-page PDF from a PIL Image via Pillow (no GS)."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    rgb.save(path)


# ── Test 1: Already-OCR'd PDF is copied ──────────────────────────────────────

class TestAlreadyOcrCopied:
    """Test 1 — Already-OCR'd PDF is copied to output directory."""

    def test_copy_preserves_file(self, tmp_path):
        src_dir = tmp_path / "source"
        out_dir = tmp_path / "output"
        src_dir.mkdir()
        out_dir.mkdir()

        pdf_path = str(src_dir / "has_text.pdf")
        _make_pdf_with_text(pdf_path)
        assert not pdf_needs_ocr(pdf_path), "PDF should already have text"

        config = _config_for(str(src_dir), str(out_dir), tmp_path)

        results = scan_directory(str(src_dir), config)
        assert len(results["ocr"]) == 1
        assert len(results["non_ocr"]) == 0

        copied = copy_ocr_pdfs(results["ocr"], config, str(src_dir))
        assert copied == 1

        dest = out_dir / "has_text.pdf"
        assert dest.exists(), "Copied file should exist in output directory"
        assert filecmp.cmp(pdf_path, str(dest), shallow=False), (
            "Copied file should be byte-identical to source"
        )

    def test_ocr_files_txt_contains_path(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        pdf_path = str(src_dir / "text.pdf")
        _make_pdf_with_text(pdf_path)

        config = _config_for(str(src_dir), tmp_path=tmp_path)

        scan_directory(str(src_dir), config)

        with open(config["ocr_output_file"]) as f:
            contents = f.read()
        assert pdf_path in contents, (
            "ocr_files.txt should contain the path"
        )

    def test_processed_count_increments(self, tmp_path):
        src_dir = tmp_path / "src"
        src_dir.mkdir()

        pdf_path = str(src_dir / "text.pdf")
        _make_pdf_with_text(pdf_path)

        config = _config_for(str(src_dir), tmp_path=tmp_path)

        results = scan_directory(str(src_dir), config)
        assert results["processed"] == 1
        assert len(results["ocr"]) == 1

    def test_no_ocr_performed(self, tmp_path):
        src_dir = tmp_path / "src"
        out_dir = tmp_path / "out"
        src_dir.mkdir()
        out_dir.mkdir()

        pdf_path = str(src_dir / "text.pdf")
        _make_pdf_with_text(pdf_path)

        config = _config_for(str(src_dir), str(out_dir), tmp_path)

        results = scan_directory(str(src_dir), config)
        copy_ocr_pdfs(results["ocr"], config, str(src_dir))

        dest = str(out_dir / "text.pdf")
        assert _check_text_layer(dest), "Copied file should still have text layer"


# ── Test 2: Clean scan — no preprocessing ───────────────────────────────────

@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestCleanScanNoPreprocessing:
    """Test 2 — Clean scan triggers no preprocessing."""

    def test_clean_pages_count_positive(self, tmp_path, monkeypatch):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        pdf_path = str(tmp_path / "clean.pdf")
        _create_img_pdf(pdf_path, img)

        monkeypatch.setattr(
            "OCRDAC.detect_preprocessing_needed",
            lambda img: (False, "none"),
        )

        output = str(tmp_path / "clean_out.pdf")
        _, _, stats = ocr_pdf_dual_image(
            pdf_path, output, "eng",
            auto_preprocessing=True,
        )

        assert stats["clean_pages"] > 0, (
            "Clean scan should have clean_pages > 0"
        )
        assert stats["preprocessed_pages"] == 0, (
            "Clean scan should have preprocessed_pages == 0"
        )
        assert len(stats["preprocessing_reasons"]) == 0, (
            "Clean scan should have no preprocessing reasons"
        )


# ── Test 3: Noisy scan triggers preprocessing ───────────────────────────────

@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestNoisyScanTriggersPreprocessing:
    """Test 3 — Noisy scan triggers preprocessing."""

    def test_low_contrast_triggers_preprocessing(self, tmp_path, monkeypatch):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        pdf_path = str(tmp_path / "noisy.pdf")
        _create_img_pdf(pdf_path, img)

        monkeypatch.setattr(
            "OCRDAC.detect_preprocessing_needed",
            lambda img: (True, "low_contrast"),
        )

        output = str(tmp_path / "noisy_out.pdf")
        _, _, stats = ocr_pdf_dual_image(
            pdf_path, output, "eng",
            auto_preprocessing=True,
        )

        assert stats["preprocessed_pages"] > 0, (
            "Should trigger preprocessing"
        )
        assert stats["clean_pages"] == 0, (
            "Preprocessed page is not clean"
        )
        assert stats["preprocessing_reasons"].get("low_contrast", 0) > 0, (
            "low_contrast reason should be recorded"
        )

    def test_stripes_triggers_preprocessing(self, tmp_path, monkeypatch):
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        pdf_path = str(tmp_path / "stripes.pdf")
        _create_img_pdf(pdf_path, img)

        monkeypatch.setattr(
            "OCRDAC.detect_preprocessing_needed",
            lambda img: (True, "stripes"),
        )

        output = str(tmp_path / "stripe_out.pdf")
        _, _, stats = ocr_pdf_dual_image(
            pdf_path, output, "eng",
            auto_preprocessing=True,
        )

        assert stats["preprocessed_pages"] > 0, (
            "Stripes should trigger preprocessing"
        )
        assert stats["preprocessing_reasons"].get("stripe_artifacts", 0) > 0, (
            "stripe_artifacts reason should be recorded"
        )


# ── Test 4: Mixed PDF (clean + noisy pages) ─────────────────────────────────

@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestMixedPdfStats:
    """Test 4 — Mixed PDF with clean and noisy pages."""

    def _make_multipage_pdf(self, pages, tmp_path):
        """Create a multi-page PDF from images via Pillow."""
        path = str(tmp_path / "multi.pdf")
        images_rgb = [p.convert("RGB") for p in pages]
        first = images_rgb[0]
        if len(images_rgb) > 1:
            first.save(path, save_all=True, append_images=images_rgb[1:])
        else:
            first.save(path)
        return path

    def test_mixed_pages_correct_counts(self, tmp_path, monkeypatch):
        clean_img = Image.new("RGB", (200, 200), (255, 255, 255))
        noisy_img = Image.new("RGB", (200, 200), (128, 128, 128))

        detections = iter([
            (False, "none"),
            (True, "low_contrast"),
        ])
        monkeypatch.setattr(
            "OCRDAC.detect_preprocessing_needed",
            lambda img: next(detections),
        )

        pdf_path = self._make_multipage_pdf([clean_img, noisy_img], tmp_path)

        output = str(tmp_path / "multi_out.pdf")
        _, _, stats = ocr_pdf_dual_image(
            pdf_path, output, "eng",
            auto_preprocessing=True,
        )

        assert stats["clean_pages"] == 1, (
            f"Expected 1 clean page, got {stats['clean_pages']}"
        )
        assert stats["preprocessed_pages"] == 1, (
            f"Expected 1 preprocessed page, got {stats['preprocessed_pages']}"
        )
        assert stats["preprocessing_reasons"].get("low_contrast", 0) >= 1, (
            "low_contrast should be recorded for the noisy page"
        )

    def test_mixed_with_stripes_and_clean(self, tmp_path, monkeypatch):
        clean_img = Image.new("RGB", (200, 200), (255, 255, 255))
        stripe_img = Image.new("RGB", (200, 200), (100, 100, 100))

        detections = iter([
            (False, "none"),
            (True, "stripes"),
        ])
        monkeypatch.setattr(
            "OCRDAC.detect_preprocessing_needed",
            lambda img: next(detections),
        )

        pdf_path = self._make_multipage_pdf([clean_img, stripe_img], tmp_path)

        output = str(tmp_path / "multi_stripe_out.pdf")
        _, _, stats = ocr_pdf_dual_image(
            pdf_path, output, "eng",
            auto_preprocessing=True,
        )

        assert stats["clean_pages"] == 1, (
            f"Expected 1 clean page, got {stats['clean_pages']}"
        )
        assert stats["preprocessed_pages"] == 1, (
            f"Expected 1 preprocessed page, got {stats['preprocessed_pages']}"
        )
        assert stats["preprocessing_reasons"].get("stripe_artifacts", 0) >= 1, (
            "stripe_artifacts should be recorded"
        )


# ── Test 5: End-to-end run with multiple files ──────────────────────────────

class TestEndToEndMultipleFiles:
    """Test 5 — End-to-end with already-OCR'd, clean, and noisy files."""

    def test_all_files_in_output_directory(self, tmp_path, monkeypatch):
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()

        ocrd_pdf = str(src / "already_ocrd.pdf")
        _make_pdf_with_text(ocrd_pdf)

        blank_pdf = str(src / "needs_ocr.pdf")
        _make_blank_pdf(blank_pdf)

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: (
                "success", "",
                {"clean_pages": 1, "preprocessed_pages": 0, "preprocessing_reasons": {}},
            ),
        )

        config = _config_for(str(src), str(out), tmp_path)

        results = scan_directory(str(src), config)
        assert results["processed"] == 2
        assert len(results["ocr"]) == 1
        assert len(results["non_ocr"]) == 1

        copy_ocr_pdfs(results["ocr"], config, str(src))

        assert (out / "already_ocrd.pdf").exists()
        assert filecmp.cmp(ocrd_pdf, str(out / "already_ocrd.pdf"), shallow=False)

        convert_results = convert_pdfs(results["non_ocr"], config, str(src))

        assert convert_results["success"] == 1, (
            "Non-OCR file should report success"
        )

    def test_ocr_and_non_ocr_files_txt(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()

        ocrd_pdf = str(src / "ocrd.pdf")
        _make_pdf_with_text(ocrd_pdf)

        blank_pdf = str(src / "needs_ocr.pdf")
        _make_blank_pdf(blank_pdf)

        config = _config_for(str(src), tmp_path=tmp_path)

        results = scan_directory(str(src), config)

        with open(config["ocr_output_file"]) as f:
            ocr_contents = f.read()
        with open(config["non_ocr_output_file"]) as f:
            non_ocr_contents = f.read()

        assert ocrd_pdf in ocr_contents
        assert blank_pdf in non_ocr_contents

    def test_summary_includes_preprocessing_stats(self, tmp_path, capsys, monkeypatch):
        src = tmp_path / "src"
        src.mkdir()

        blank_pdf = str(src / "test.pdf")
        _make_blank_pdf(blank_pdf)

        monkeypatch.setattr(
            "OCRDAC.ocr_pdf_dual_image",
            lambda *a, **kw: (
                "success", "",
                {"clean_pages": 2, "preprocessed_pages": 1,
                 "preprocessing_reasons": {"low_contrast": 1}},
            ),
        )

        config = _config_for(str(src), tmp_path=tmp_path)
        config["preserve_original_pdf"] = "false"

        results = scan_directory(str(src), config)
        convert_results = convert_pdfs(results["non_ocr"], config, str(src))

        print_summary(results, convert_results, 5)
        captured = capsys.readouterr()

        assert "Auto-Preprocessing Summary" in captured.out
        assert "Clean pages (no preprocessing needed): 2" in captured.out
        assert "Preprocessed pages: 1" in captured.out
        assert "low_contrast: 1" in captured.out
