"""Tests for the dual-image OCR pipeline — image integrity and preprocessing."""

import os
import sys
import subprocess
import tempfile

import pikepdf
import pytest
from PIL import Image, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import (
    preprocess_ocr_image,
    render_pages_to_images,
    ocr_pdf_dual_image,
    read_metadata,
)
from preprocessing_detector import detect_preprocessing_needed


def _make_test_pdf(path, pages=1, width=612, height=792, color=(255, 255, 255)):
    """Create a minimal PDF with one or more blank pages."""
    pdf = pikepdf.Pdf.new()
    for _ in range(pages):
        pdf.add_blank_page(page_size=(width, height))
    pdf.save(path)
    pdf.close()


def _make_image_with_content(width=200, height=200):
    """Create a simple image with some text-like content."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    pixels = img.load()
    for y in range(40, 46):
        for x in range(30, 170):
            pixels[x, y] = (0, 0, 0)
    for y in range(80, 86):
        for x in range(30, 150):
            pixels[x, y] = (0, 0, 0)
    for y in range(120, 126):
        for x in range(30, 160):
            pixels[x, y] = (0, 0, 0)
    return img


# ── Preprocessing Integrity Tests ────────────────────────────────────────────

class TestImageOriginalUnchanged:
    def test_original_unchanged_after_copy(self):
        img = _make_image_with_content()
        original_pixels = list(img.getdata())
        copy = img.copy()
        preprocess_ocr_image(copy, 3, 130)
        assert list(img.getdata()) == original_pixels

    def test_original_unchanged_after_detection(self):
        img = _make_image_with_content()
        original_pixels = list(img.getdata())
        needs, reason = detect_preprocessing_needed(img)
        assert list(img.getdata()) == original_pixels

    def test_copy_is_separate_object(self):
        img = _make_image_with_content()
        copy = img.copy()
        preprocess_ocr_image(copy, 3, 130)
        assert copy is not img
        assert copy.size == img.size


class TestPreprocessingAppliedOnlyWhenNeeded:
    def test_preprocessing_modifies_copy(self):
        from OCRDAC import preprocess_ocr_image
        img = _make_image_with_content()
        original_data = img.tobytes()
        processed = preprocess_ocr_image(img, 3, 130)
        assert processed.tobytes() != original_data

    def test_preprocessing_output_is_rgb(self):
        from OCRDAC import preprocess_ocr_image
        img = _make_image_with_content()
        processed = preprocess_ocr_image(img, 3, 130)
        assert processed.mode == "RGB"

    def test_no_preprocessing_preserves_image(self):
        img = _make_image_with_content()
        copy = img.copy()
        ocr_img = copy.convert("RGB")
        assert ocr_img.tobytes() == img.convert("RGB").tobytes()


class TestDetectorOnSynthetic:
    def test_low_contrast_detected(self):
        from tests.test_preprocessing_detector import _make_low_contrast_image
        img = _make_low_contrast_image()
        needs, reason = detect_preprocessing_needed(img)
        assert needs is True
        assert reason == "low_contrast"

    def test_stripes_detected(self):
        from tests.test_preprocessing_detector import _make_striped_image
        img = _make_striped_image()
        needs, reason = detect_preprocessing_needed(img)
        assert needs is True
        assert reason == "stripes"

    def test_clean_scan_not_detected(self):
        from tests.test_preprocessing_detector import _make_clean_scan_image
        img = _make_clean_scan_image()
        needs, reason = detect_preprocessing_needed(img)
        assert needs is False
        assert reason == "none"


# ── OCR Pipeline Integration (requires ocrmypdf + gs + tesseract) ───────────

@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestDualImagePipeline:
    @pytest.fixture
    def simple_pdf(self, tmp_path):
        path = str(tmp_path / "input.pdf")
        _make_test_pdf(path, pages=1)
        return path

    def test_pipeline_returns_success(self, simple_pdf, tmp_path):
        output = str(tmp_path / "output.pdf")
        status, msg, _ = ocr_pdf_dual_image(
            simple_pdf, output, "eng",
            auto_preprocessing=True,
            preprocessing_setting="none",
            median_filter_size=3,
            threshold_val=130,
            ocrdac_version="v0.3",
        )
        assert status == "success", f"Pipeline failed: {msg}"

    def test_output_pdf_has_correct_metadata(self, simple_pdf, tmp_path):
        output = str(tmp_path / "output.pdf")
        ocr_pdf_dual_image(
            simple_pdf, output, "eng",
            auto_preprocessing=True,
        )
        meta = read_metadata(output)
        assert "OCRDAC-Version" in meta
        assert "Auto-Preprocessing-Enabled" in meta
        assert "OCRmyPDF-Version" in meta
        assert "Ghostscript-Version" in meta
        assert meta["OCRDAC-Version"] == "v0.4"

    def test_output_pdf_is_valid(self, simple_pdf, tmp_path):
        output = str(tmp_path / "output.pdf")
        ocr_pdf_dual_image(
            simple_pdf, output, "eng",
            auto_preprocessing=True,
        )
        with pikepdf.Pdf.open(output) as pdf:
            assert len(pdf.pages) > 0

    def test_pipeline_reprocess_same_output(self, tmp_path):
        """Verify pipeline can process a PDF and then re-process its output."""
        from PIL import Image
        # Use the sample PDF from the project
        sample = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "original_pdf_files",
            "2005-10-31 Recall - Replace Steering shaft.pdf"
        )
        if not os.path.exists(sample):
            pytest.skip("Sample PDF not found")

        output = str(tmp_path / "out1.pdf")
        status, msg, _ = ocr_pdf_dual_image(sample, output, "eng")
        assert status == "success", f"First run failed: {msg}"

        # Verify output is valid
        with pikepdf.Pdf.open(output) as pdf:
            assert len(pdf.pages) > 0

        # Second run on output should succeed (and typically skip)
        output2 = str(tmp_path / "out2.pdf")
        status2, msg2, _ = ocr_pdf_dual_image(output, output2, "eng")
        assert status2 in ("success", "skipped"), f"Second run failed: {msg2}"
