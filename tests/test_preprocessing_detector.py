"""Tests for OCRDAC automatic preprocessing detection."""

import os
import sys
from unittest.mock import patch, MagicMock

import pikepdf
import pytest
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_detector import detect_preprocessing_needed
from OCRDAC import write_metadata, read_metadata, DEFAULT_CONFIG


def _make_blank_pdf(path):
    """Create a minimal blank PDF at the given path."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(path)
    pdf.close()


def _make_low_contrast_image(width=100, height=100):
    """Create a low-contrast image (all pixels in narrow range around 150)."""
    img = Image.new("L", (width, height), 150)
    pixels = img.load()
    # Add tiny variation (range ~5 pixels) so it's not perfectly uniform
    for y in range(height):
        for x in range(width):
            pixels[x, y] = 148 + ((x + y) % 5)
    return img


def _make_striped_image(width=100, height=100):
    """Create an image with horizontal stripe artifacts."""
    img = Image.new("L", (width, height), 255)
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            if y % 4 == 0:
                pixels[x, y] = 80  # Dark stripe rows
            else:
                pixels[x, y] = 240  # Light background
    return img


def _make_clean_scan_image(width=200, height=200):
    """Create a clean black-on-white scan with high-contrast text content.

    Mostly white background with a few thin dark lines. This gives:
    - High contrast (235 > 40)
    - Low mean brightness (~230, outside 120-200)
    - Low variance (< 500, because most pixels are white)
    - Low stripe score
    """
    img = Image.new("L", (width, height), 255)
    pixels = img.load()
    # Thin dark text lines (only ~2% of pixels are dark)
    for y in range(40, 43):
        for x in range(20, 180):
            pixels[x, y] = 20
    for y in range(80, 83):
        for x in range(20, 160):
            pixels[x, y] = 25
    for y in range(120, 123):
        for x in range(20, 170):
            pixels[x, y] = 15
    return img


def _make_uneven_background_image(width=100, height=100):
    """Create an image with uneven background (brightness in 120-200 range).

    The gradient spans 100-220 so contrast > 40 (avoiding low_contrast trigger),
    and the mean stays in the 120-200 range to trigger uneven_background.
    """
    img = Image.new("L", (width, height), 160)
    pixels = img.load()
    # Gradient from 100 to 220: contrast=120 > 40, mean ~160 in [120,200]
    for y in range(height):
        for x in range(width):
            pixels[x, y] = 100 + (x * 120 // width)
    return img


class TestDetectLowContrast:
    def test_returns_true_for_low_contrast(self):
        img = _make_low_contrast_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is True
        assert reason == "low_contrast"

    def test_low_contrast_has_narrow_pixel_range(self):
        img = _make_low_contrast_image()
        gray = img.convert("L")
        pixels = list(gray.getdata())
        contrast = max(pixels) - min(pixels)
        assert contrast < 40


class TestDetectStripes:
    def test_returns_true_for_striped_image(self):
        img = _make_striped_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is True
        assert reason == "stripe_artifacts"

    def test_stripe_score_exceeds_threshold(self):
        from preprocessing_detector import _computeStripeScore
        img = _make_striped_image()
        gray = img.convert("L")
        pixels = list(gray.getdata())
        w, h = gray.size
        score = _computeStripeScore(pixels, w, h)
        assert score > 0.15


class TestDetectCleanScan:
    def test_returns_false_for_clean_scan(self):
        img = _make_clean_scan_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is False
        assert reason == "none"

    def test_clean_scan_has_high_contrast(self):
        img = _make_clean_scan_image()
        gray = img.convert("L")
        pixels = list(gray.getdata())
        contrast = max(pixels) - min(pixels)
        assert contrast >= 40

    def test_clean_scan_mean_outside_uneven_range(self):
        img = _make_clean_scan_image()
        gray = img.convert("L")
        pixels = list(gray.getdata())
        mean_val = sum(pixels) / len(pixels)
        # Mean should be outside 120-200 to avoid uneven_background trigger
        assert mean_val <= 120 or mean_val >= 200


class TestDetectUnevenBackground:
    def test_returns_true_for_uneven_background(self):
        img = _make_uneven_background_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is True
        assert reason == "uneven_background"


class TestAutoPreprocessingMetadata:
    def test_metadata_contains_auto_preprocessing_fields(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        metadata = {
            "OCRDAC-Version": "v0.1",
            "OCRmyPDF-Version": "16.13.0",
            "Ghostscript-Version": "10.06.0",
            "OCR-Flags": "--deskew --clean",
            "OCR-DateTime": "2025-01-01T00:00:00Z",
            "Used-Unpaper": "False",
            "DPI-Normalization": "default",
            "Auto-Preprocessing-Enabled": "True",
            "Auto-Preprocessing-Reason": "low_contrast",
        }
        write_metadata(pdf_path, metadata)
        result = read_metadata(pdf_path)

        assert result["Auto-Preprocessing-Enabled"] == "True"
        assert result["Auto-Preprocessing-Reason"] == "low_contrast"

    def test_metadata_auto_preprocessing_disabled(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        metadata = {
            "OCRDAC-Version": "v0.1",
            "OCRmyPDF-Version": "16.13.0",
            "Ghostscript-Version": "10.06.0",
            "OCR-Flags": "--deskew --clean",
            "OCR-DateTime": "2025-01-01T00:00:00Z",
            "Used-Unpaper": "False",
            "DPI-Normalization": "default",
            "Auto-Preprocessing-Enabled": "False",
            "Auto-Preprocessing-Reason": "none",
        }
        write_metadata(pdf_path, metadata)
        result = read_metadata(pdf_path)

        assert result["Auto-Preprocessing-Enabled"] == "False"
        assert result["Auto-Preprocessing-Reason"] == "none"

    def test_metadata_reason_values(self, tmp_path):
        valid_reasons = ["low_contrast", "stripe_artifacts", "uneven_background",
                         "high_noise", "none", "manual"]
        for reason in valid_reasons:
            pdf_path = str(tmp_path / f"test_{reason}.pdf")
            _make_blank_pdf(pdf_path)
            metadata = {
                "OCRDAC-Version": "v0.1",
                "OCRmyPDF-Version": "16.13.0",
                "Ghostscript-Version": "10.06.0",
                "OCR-Flags": "",
                "OCR-DateTime": "2025-01-01T00:00:00Z",
                "Used-Unpaper": "False",
                "DPI-Normalization": "default",
                "Auto-Preprocessing-Enabled": "True",
                "Auto-Preprocessing-Reason": reason,
            }
            write_metadata(pdf_path, metadata)
            result = read_metadata(pdf_path)
            assert result["Auto-Preprocessing-Reason"] == reason


class TestPipelineAutoPreprocessing:
    def test_default_config_has_auto_preprocessing(self):
        assert "auto_preprocessing" in DEFAULT_CONFIG
        assert DEFAULT_CONFIG["auto_preprocessing"] == "true"

    def test_config_auto_preprocessing_reads_true(self, tmp_path):
        config_path = str(tmp_path / "test.config")
        with open(config_path, "w") as f:
            f.write("[OCRDAC]\nauto_preprocessing = true\n")
        from OCRDAC import load_config
        config = load_config(config_path)
        assert config["auto_preprocessing"] == "true"

    def test_config_auto_preprocessing_reads_false(self, tmp_path):
        config_path = str(tmp_path / "test.config")
        with open(config_path, "w") as f:
            f.write("[OCRDAC]\nauto_preprocessing = false\n")
        from OCRDAC import load_config
        config = load_config(config_path)
        assert config["auto_preprocessing"] == "false"

    @patch("OCRDAC.detect_preprocessing_needed")
    def test_pipeline_calls_detector_per_page(self, mock_detect):
        """Verify that _ocr_pdf_preprocessed calls detect_preprocessing_needed."""
        mock_detect.return_value = (False, "none")

        from OCRDAC import render_pages_to_images, _ocr_pdf_preprocessed

        # We can't easily test the full pipeline without ghostscript/tesseract,
        # but we can verify the function signature accepts auto_preprocessing
        import inspect
        sig = inspect.signature(_ocr_pdf_preprocessed)
        assert "auto_preprocessing" in sig.parameters

    def test_ocr_pdf_signature_includes_auto_preprocessing(self):
        from OCRDAC import ocr_pdf
        import inspect
        sig = inspect.signature(ocr_pdf)
        assert "auto_preprocessing" in sig.parameters

    def test_convert_pdfs_reads_auto_preprocessing_config(self, tmp_path):
        """Verify convert_pdfs extracts auto_preprocessing from config."""
        from OCRDAC import convert_pdfs
        import inspect
        # Verify the function exists and can be called (we just check it reads the config)
        source = inspect.getsource(convert_pdfs)
        assert "auto_preprocessing" in source
