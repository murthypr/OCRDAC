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
                pixels[x, y] = 80
            else:
                pixels[x, y] = 240
    return img


def _make_clean_scan_image(width=200, height=200):
    """Create a clean black-on-white scan with high-contrast text content."""
    img = Image.new("L", (width, height), 255)
    pixels = img.load()
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
    """Create an image with uneven background (brightness in 120-200 range)."""
    img = Image.new("L", (width, height), 160)
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = 100 + (x * 120 // width)
    return img


# ── Synthetic Low-Contrast Image Tests ─────────────────────────────────────

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


# ── Synthetic Striped Image Tests ──────────────────────────────────────────

class TestDetectStripes:
    def test_returns_true_for_striped_image(self):
        img = _make_striped_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is True
        assert reason == "stripes"

    def test_stripe_score_exceeds_threshold(self):
        from preprocessing_detector import _computeStripeScore
        img = _make_striped_image()
        gray = img.convert("L")
        pixels = list(gray.getdata())
        w, h = gray.size
        score = _computeStripeScore(pixels, w, h)
        assert score > 0.15


# ── Clean Scan Image Tests ─────────────────────────────────────────────────

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
        assert mean_val <= 120 or mean_val >= 200


# ── Uneven Background Image Tests ──────────────────────────────────────────

class TestDetectUnevenBackground:
    def test_returns_true_for_uneven_background(self):
        img = _make_uneven_background_image()
        needs_preprocessing, reason = detect_preprocessing_needed(img)
        assert needs_preprocessing is True
        assert reason == "uneven_background"
