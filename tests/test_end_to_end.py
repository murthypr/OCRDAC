"""End-to-end tests for OCRDAC v2 dual-image pipeline.

These tests create synthetic PDFs with specific characteristics and verify:
1. Output PDF is visually identical to input (pixel-perfect preservation)
2. OCR text layer is present and searchable (using the real sample PDF)
3. Auto-preprocessing triggers correctly
4. Metadata is correct
5. No conflicting flags are used
6. No blank pages or missing text

Requires: ocrmypdf, ghostscript, tesseract, pikepdf, Pillow
"""

import os
import sys
import subprocess
import tempfile

import pikepdf
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import (
    ocr_pdf_dual_image,
    read_metadata,
    render_pages_to_images,
    preprocess_ocr_image,
)


SAMPLE_PDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "original_pdf_files",
    "2005-10-31 Recall - Replace Steering shaft.pdf",
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _create_pdf_from_images(images, path):
    """Create a multi-page PDF from a list of PIL Images via GS."""
    png_paths = []
    for i, img in enumerate(images):
        p = path.replace(".pdf", f"_{i:03d}.png")
        img.save(p)
        png_paths.append(p)

    subprocess.run(
        ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
         "-dPDFSETTINGS=/prepress",
         f"-sOutputFile={path}"] + png_paths,
        capture_output=True, timeout=120
    )


def _render_to_pixels(pdf_path, dpi=300, page=0):
    """Render a PDF page to a PIL Image for comparison.
    Uses 300 DPI by default so each output pixel maps 1:1 to the
    original source pixels (no downscaling interpolation)."""
    images = render_pages_to_images(pdf_path, dpi=dpi)
    if page < len(images):
        return images[page]
    return None


def _pixel_difference(img1, img2):
    """Compute the maximum pixel difference between two images."""
    if img1.size != img2.size:
        return float('inf')
    if img1.mode != img2.mode:
        img2 = img2.convert(img1.mode)
    p1 = list(img1.getdata())
    p2 = list(img2.getdata())
    max_diff = 0
    for a, b in zip(p1, p2):
        if isinstance(a, tuple):
            diff = max(abs(a[i] - b[i]) for i in range(len(a)))
        else:
            diff = abs(a - b)
        max_diff = max(max_diff, diff)
    return max_diff


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


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_pdf(tmp_path_factory):
    """Create a high-contrast synthetic PDF for visual comparison tests."""
    tmpdir = tmp_path_factory.mktemp("clean_pdf")
    width, height = 800, 1000
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for y in range(100, 900, 100):
        for dx in range(100, 700, 20):
            for dy in range(12):
                for ddx in range(10):
                    px, py = dx + ddx, y + dy
                    if px < width and py < height:
                        draw.point((px, py), fill=(0, 0, 0))

    path = str(tmpdir / "clean.pdf")
    _create_pdf_from_images([img], path)
    return path


@pytest.fixture(scope="module")
def grey_background_pdf(tmp_path_factory):
    """Create a PDF with grey background for preprocessing trigger tests."""
    tmpdir = tmp_path_factory.mktemp("grey_pdf")
    width, height = 400, 500
    bg = 180
    img = Image.new("RGB", (width, height), (bg, bg, bg))
    draw = ImageDraw.Draw(img)
    for y in range(50, 450, 60):
        for dx in range(30, 350, 15):
            for dy in range(8):
                for ddx in range(6):
                    px, py = dx + ddx, y + dy
                    if px < width and py < height:
                        draw.point((px, py), fill=(bg - 20, bg - 20, bg - 20))

    path = str(tmpdir / "grey.pdf")
    _create_pdf_from_images([img], path)
    return path


@pytest.fixture(scope="module")
def stripe_pdf(tmp_path_factory):
    """Create a PDF with horizontal stripe artifacts."""
    tmpdir = tmp_path_factory.mktemp("stripe_pdf")
    width, height = 400, 500
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for y in range(0, height, 6):
        for x in range(width):
            draw.point((x, y), fill=(180, 180, 180))

    for y in range(50, 450, 80):
        for dx in range(30, 350, 15):
            for dy in range(8):
                for ddx in range(6):
                    px, py = dx + ddx, y + dy
                    if px < width and py < height:
                        draw.point((px, py), fill=(0, 0, 0))

    path = str(tmpdir / "stripe.pdf")
    _create_pdf_from_images([img], path)
    return path


# ── End-to-End Tests ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not os.path.exists(SAMPLE_PDF),
    reason="Sample PDF not found",
)
@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestEndToEndWithRealSample:

    def test_sample_pdf_visually_crisp(self, tmp_path):
        output = str(tmp_path / "sample_out.pdf")
        status, msg = ocr_pdf_dual_image(SAMPLE_PDF, output, "eng")
        assert status == "success", f"Pipeline failed: {msg}"

        # Compare at native 300 DPI (no downscaling interpolation)
        orig_img = _render_to_pixels(SAMPLE_PDF)
        out_img = _render_to_pixels(output)
        assert orig_img is not None
        assert out_img is not None
        diff = _pixel_difference(orig_img, out_img)
        assert diff == 0, f"Visual layer differs! Max pixel diff: {diff}"

    def test_sample_pdf_has_text_layer(self, tmp_path):
        output = str(tmp_path / "sample_ocr.pdf")
        ocr_pdf_dual_image(SAMPLE_PDF, output, "eng")
        assert _check_text_layer(output), "No text layer found in OCR output"

    def test_sample_pdf_auto_preprocessing_triggers(self, tmp_path):
        output = str(tmp_path / "sample_meta.pdf")
        ocr_pdf_dual_image(SAMPLE_PDF, output, "eng", auto_preprocessing=True)
        meta = read_metadata(output)
        assert "Auto-Preprocessing-Reason" in meta

    def test_sample_pdf_no_blank_pages(self, tmp_path):
        output = str(tmp_path / "sample_noblank.pdf")
        ocr_pdf_dual_image(SAMPLE_PDF, output, "eng")
        with pikepdf.Pdf.open(output) as pdf:
            assert len(pdf.pages) > 0
            for page in pdf.pages:
                contents = page.get("/Contents")
                assert contents is not None

    def test_metadata_has_all_required_fields(self, tmp_path):
        output = str(tmp_path / "meta_all.pdf")
        ocr_pdf_dual_image(SAMPLE_PDF, output, "eng")
        meta = read_metadata(output)
        required = [
            "OCRDAC-Version",
            "Auto-Preprocessing-Enabled",
            "Auto-Preprocessing-Reason",
            "Median-Filter-Used",
            "Median-Filter-Size",
            "Threshold-Used",
            "OCRmyPDF-Version",
            "Ghostscript-Version",
            "OCR-DateTime",
        ]
        for field in required:
            assert field in meta, f"Missing metadata field: {field}"

    def test_no_conflicting_flags(self, tmp_path, monkeypatch):
        """Verify OCRmyPDF is called without --clean, --deskew, etc."""
        original_run = subprocess.run

        def capture_run(*args, **kwargs):
            cmd = args[0] if args else kwargs.get('args', [])
            if cmd and 'ocrmypdf' in str(cmd[0]):
                cmd_str = ' '.join(cmd)
                assert '--clean' not in cmd_str, f"Conflicting flag: {cmd_str}"
                assert '--deskew' not in cmd_str, f"Conflicting flag: {cmd_str}"
                assert '--remove-background' not in cmd_str, f"Conflicting flag: {cmd_str}"
                assert '--rotate-pages' not in cmd_str, f"Conflicting flag: {cmd_str}"
                assert '--optimize' not in cmd_str, f"Conflicting flag: {cmd_str}"
            return original_run(*args, **kwargs)

        monkeypatch.setattr(subprocess, 'run', capture_run)
        output = str(tmp_path / "no_conflict.pdf")
        status, msg = ocr_pdf_dual_image(SAMPLE_PDF, output, "eng")
        assert status == "success", f"Pipeline failed: {msg}"


@pytest.mark.skipif(
    not any(
        os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
        for p in os.environ["PATH"].split(os.pathsep)
    ),
    reason="ocrmypdf not installed",
)
class TestEndToEndSynthetic:

    def test_clean_pdf_visually_identical(self, clean_pdf, tmp_path):
        output = str(tmp_path / "clean_out.pdf")
        status, msg = ocr_pdf_dual_image(
            clean_pdf, output, "eng",
            auto_preprocessing=True,
        )
        assert status == "success", f"Pipeline failed: {msg}"

        orig_img = _render_to_pixels(clean_pdf, dpi=72)
        out_img = _render_to_pixels(output, dpi=72)
        diff = _pixel_difference(orig_img, out_img)
        assert diff == 0, f"Pixel difference: {diff}"

    def test_clean_pdf_no_blank_pages(self, clean_pdf, tmp_path):
        output = str(tmp_path / "clean_noblank.pdf")
        ocr_pdf_dual_image(clean_pdf, output, "eng")
        with pikepdf.Pdf.open(output) as pdf:
            assert len(pdf.pages) > 0

    def test_grey_pdf_visually_identical(self, grey_background_pdf, tmp_path):
        output = str(tmp_path / "grey_out.pdf")
        status, msg = ocr_pdf_dual_image(
            grey_background_pdf, output, "eng",
            auto_preprocessing=True,
        )
        assert status == "success", f"Pipeline failed: {msg}"

        orig_img = _render_to_pixels(grey_background_pdf, dpi=72)
        out_img = _render_to_pixels(output, dpi=72)
        diff = _pixel_difference(orig_img, out_img)
        assert diff == 0, f"Pixel difference: {diff}"

    def test_grey_pdf_preprocessing_triggered(self, grey_background_pdf, tmp_path):
        output = str(tmp_path / "grey_meta.pdf")
        ocr_pdf_dual_image(grey_background_pdf, output, "eng", auto_preprocessing=True)
        meta = read_metadata(output)
        assert meta["Median-Filter-Used"] == "True"

    def test_stripe_pdf_visually_identical(self, stripe_pdf, tmp_path):
        output = str(tmp_path / "stripe_out.pdf")
        status, msg = ocr_pdf_dual_image(
            stripe_pdf, output, "eng",
            auto_preprocessing=True,
        )
        assert status == "success", f"Pipeline failed: {msg}"

        orig_img = _render_to_pixels(stripe_pdf, dpi=72)
        out_img = _render_to_pixels(output, dpi=72)
        diff = _pixel_difference(orig_img, out_img)
        assert diff == 0, f"Pixel difference: {diff}"

    def test_stripe_pdf_preprocessing_triggered(self, stripe_pdf, tmp_path):
        output = str(tmp_path / "stripe_meta.pdf")
        ocr_pdf_dual_image(stripe_pdf, output, "eng", auto_preprocessing=True)
        meta = read_metadata(output)
        # Preprocessing must be used (the detector may flag as stripes or
        # low_contrast depending on GS anti-aliasing)
        assert meta["Median-Filter-Used"] == "True"
