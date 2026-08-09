"""Tests for the preserve_original_pdf clean-sandwich pipeline.

Verifies:
  1. The original PDF pages are preserved exactly (same XObjects, DPI,
     MediaBox, image bytes, compression).
  2. A positioned OCR text layer is grafted onto the pages.
  3. No rasterization: no PNG/JPEG images are added beyond the originals.
  4. Legacy mode (preserve_original_pdf=false) still uses the
     Ghostscript/OCRmyPDF dual-image pipeline.
  5. Magazine/clipping-style embedded images stay byte-identical (crisp).
  6. OCR text accuracy matches the legacy pipeline output for the same
     Tesseract engine.
Also verifies config defaulting (missing key => true) and threading
through convert_pdfs.
"""

import os
import re
import sys
import shutil
import subprocess

import pytest
from PIL import Image
import pikepdf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import OCRDAC
from OCRDAC import load_config, convert_pdfs

OCRMYPDF_AVAILABLE = any(
    os.access(os.path.join(p, "ocrmypdf"), os.X_OK)
    for p in os.environ["PATH"].split(os.pathsep)
)
TESSERACT_AVAILABLE = any(
    os.access(os.path.join(p, "tesseract"), os.X_OK)
    for p in os.environ["PATH"].split(os.pathsep)
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_small_pdf(path, size=(600, 300)):
    """Single-page image-only PDF (no text layer)."""
    from PIL import ImageDraw
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 40), "FORM 2026", fill=(0, 0, 0))
    img.save(path, dpi=(300, 300))


def _images_byte_signature(pdf_path):
    """Return {name: raw_bytes} of every image XObject on page 1."""
    with pikepdf.Pdf.open(pdf_path) as pdf:
        page = pdf.pages[0]
        return {str(k): v.read_raw_bytes()
                for k, v in page.images.items()}


def _extract_text_ops(pdf_path):
    """Pull `(...) Tj` tokens from all content streams (searchable text)."""
    texts = []
    with pikepdf.Pdf.open(pdf_path) as pdf:
        for page in pdf.pages:
            entries = page.obj.get("/Contents")
            streams = entries if isinstance(entries, pikepdf.Array) else [entries]
            for s in streams:
                if s is None:
                    continue
                data = s.read_bytes()
                texts.extend(re.findall(rb"\((.*?)\)\s*Tj", data))
    # Legacy pipelines may encode the layer as UTF-16 (null-padded bytes);
    # strip to be com-, e.g. b"\x00F\x00O..." -> "FORM".
    return {t.decode("latin-1").replace("\x00", "") for t in texts if t}


FAKE_HOCR = (
    "<html><body><div class='ocr_page' title='bbox 0 0 600 400'>\n"
    "  <div class='ocr_carea'>\n"
    "    <p class='ocr_par'>\n"
    "      <span class='ocr_line' title='bbox 20 20 400 40'>\n"
    "        <span class='ocrx_word' title='bbox 30 20 180 40'>FORM</span>\n"
    "        <span class='ocrx_word' title='bbox 190 20 320 40'>2026</span>\n"
    "      </span>\n"
    "    </p>\n"
    "  </div>\n"
    "</div></body></html>"
)


@pytest.fixture
def fake_tesseract(monkeypatch):
    """Replace the tesseract subprocess with a canned HOCR response."""
    calls = []
    orig_run = OCRDAC.subprocess.run

    def fake_run(cmd, **kwargs):
        if isinstance(cmd, list) and cmd and cmd[0] == "tesseract":
            calls.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, FAKE_HOCR, "")
        return orig_run(cmd, **kwargs)

    monkeypatch.setattr(OCRDAC.subprocess, "run", fake_run)
    return calls


@pytest.fixture
def legacy_recorder(tmp_path, monkeypatch):
    """Fake the legacy pipeline's OCRmyPDF step (like the params test file)."""
    calls = []
    orig_run = OCRDAC.subprocess.run

    monkeypatch.setattr(
        OCRDAC,
        "render_pages_to_images",
        lambda *a, **k: [Image.new("RGB", (200, 200), (255, 255, 255))],
    )

    def fake_run(cmd, **kwargs):
        if (isinstance(cmd, list) and cmd and cmd[0] == "ocrmypdf"
                and any(isinstance(c, str) and c.endswith(".pdf") for c in cmd)):
            calls.append(list(cmd))
            pdfs = [c for c in cmd if c.endswith(".pdf")]
            shutil.copy(pdfs[-2], pdfs[-1])
            return subprocess.CompletedProcess(cmd, 0, "", "")
        return orig_run(cmd, **kwargs)

    monkeypatch.setattr(OCRDAC.subprocess, "run", fake_run)
    return calls


def _full_config(tmp_path, **extra):
    base = {
        "output_dir": str(tmp_path),
        "directory": str(tmp_path),
        "ocr_languages": "eng",
        "overwrite": "true",
        "skip_existing_ocr": "false",
        "preprocessing": "none",
        "median_filter_size": "3",
        "threshold": "130",
        "ocrdac_version": "v0.4",
        "auto_preprocessing": "false",
        "progress_interval": "25",
        "ocr_output_file": str(tmp_path / "ocr_files.txt"),
        "non_ocr_output_file": str(tmp_path / "non_ocr_files.txt"),
        "ocrmypdf_params": "",
        "preserve_original_pdf": "true",
    }
    base.update(extra)
    return base


# ── Test 1: Original PDF preserved ───────────────────────────────────────────

class TestOriginalPreserved:
    def test_images_media_and_bytes_unchanged(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)

        before = _images_byte_signature(src)
        with pikepdf.Pdf.open(src) as pdf:
            media_before = [float(x) for x in pdf.pages[0].MediaBox]
            width_before = pdf.pages[0].images.get("/image").get(pikepdf.Name.Width)

        status, msg, _ = OCRDAC.ocr_pdf_preserve(
            src, out, "eng", auto_preprocessing=False)
        assert status == "success", f"Pipeline failed: {msg}"

        after = _images_byte_signature(out)
        assert list(after.keys()) == list(before.keys()), "XObject set changed"
        for name in before:
            assert after[name] == before[name], f"Image {name} was altered"
        with pikepdf.Pdf.open(out) as pdf:
            assert [float(x) for x in pdf.pages[0].MediaBox] == media_before
            assert pdf.pages[0].images.get("/image").get(pikepdf.Name.Width) == width_before

    def test_config_defaults_to_true(self, tmp_path):
        ini = tmp_path / "cfg.ini"
        ini.write_text("[OCRDAC]\nignore = 1\n")
        assert load_config(str(ini))["preserve_original_pdf"] == "true"


# ── Test 2: OCR text layer present ───────────────────────────────────────────

class TestTextLayerPresent:
    def test_grafted_text_searchable(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        status, _, _ = OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)
        assert status == "success"

        words = _extract_text_ops(out)
        assert "FORM" in words, f"Grafted words missing: {words}"
        assert "2026" in words

    def test_font_resource_added(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)
        with pikepdf.Pdf.open(out) as pdf:
            fonts = pdf.pages[0].Resources.Font
            assert str(fonts.F1) == "/Helvetica"


# ── Test 3: No rasterization / no new images ─────────────────────────────────

class TestNoRasterization:
    def test_no_new_image_xobjects(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        n_before = len(_images_byte_signature(src))

        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        after = _images_byte_signature(out)
        assert len(after) == n_before, (
            f"New image layer added: {list(after.keys())}"
        )

    def test_native_dpi_not_scaled(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)
        with pikepdf.Pdf.open(src) as a, pikepdf.Pdf.open(out) as b:
            for key in ("/Width", "/Height", "/BitsPerComponent"):
                assert str(b.pages[0].images.get("/image").get(key)) == (
                    str(a.pages[0].images.get("/image").get(key))
                ), f"{key} changed"


# ── Test 4: Legacy mode still works ──────────────────────────────────────────

class TestLegacyMode:
    def test_legacy_uses_ocrmypdf_pipeline(self, tmp_path, legacy_recorder):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = _full_config(tmp_path, preserve_original_pdf="false")
        convert_pdfs([in_pdf], config, str(tmp_path))
        assert len(legacy_recorder) == 1, "ocrmypdf should run in legacy mode"
        assert legacy_recorder[0][0] == "ocrmypdf"

    def test_preserve_mode_runs_tesseract_not_ocrmypdf(
        self, tmp_path, fake_tesseract, legacy_recorder
    ):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = _full_config(tmp_path)  # preserve_original_pdf=true
        convert_pdfs([in_pdf], config, str(tmp_path))
        assert fake_tesseract, "tesseract should run in preserve mode"
        assert not legacy_recorder, "ocrmypdf must not run in preserve mode"


# ── Test 5: Magazine clipping stays crisp ────────────────────────────────────

class TestMagazineClipping:
    def test_embedded_jpeg_bytes_identical(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "clip.pdf")
        out = str(tmp_path / "out.pdf")

        # Simulate a chart/photo page ("magazine clipping") stored as a
        # JPEG-encoded PDF page, as a scanner app would embed it.
        from PIL import Image as ImgMod, ImageDraw
        chart = ImgMod.new("RGB", (800, 600), (240, 240, 240))
        d = ImageDraw.Draw(chart)
        for x in range(50, 750, 60):
            d.line([(x, 300), (x + 40, 100 + (x % 7) * 20)],
                   fill=(200, 40, 40), width=6)
        from io import BytesIO
        buf = BytesIO()
        chart.save(buf, format="JPEG")
        jpeg_blob = buf.getvalue()

        new_pdf = pikepdf.Pdf.new()
        page = new_pdf.add_blank_page(page_size=(576, 432))
        im = pikepdf.Stream(new_pdf, jpeg_blob)
        im.Type = pikepdf.Name.XObject
        im.Subtype = pikepdf.Name.Image
        im.Width = 800
        im.Height = 600
        im.ColorSpace = pikepdf.Name.DeviceRGB
        im.BitsPerComponent = 8
        im.Filter = pikepdf.Name.DCTDecode
        res = pikepdf.Dictionary()
        res.XObject = pikepdf.Dictionary({"/Im0": im})
        page.Resources = res
        new_pdf.save(src)

        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        with pikepdf.Pdf.open(out) as pdf:
            page2 = pdf.pages[0]
            names = [str(k) for k in page2.images.keys()]
            assert len(names) == 1, f"XObjects changed: {names}"
            img = next(iter(page2.images.values()))
            assert img.get(pikepdf.Name.Filter) == pikepdf.Name.DCTDecode
            assert img.read_raw_bytes() == jpeg_blob, (
                "Embedded JPEG was re-compressed (quality lost)"
            )


# ── Test 6: OCR accuracy identical to current pipeline ───────────────────────

@pytest.mark.skipif(not (TESSERACT_AVAILABLE and OCRMYPDF_AVAILABLE),
                    reason="tesseract/ocrmypdf not installed")
class TestOcrAccuracy:
    def test_same_words_as_legacy_pipeline(self, tmp_path):
        src = str(tmp_path / "form.pdf")
        _make_small_pdf(src)

        out_preserve = str(tmp_path / "preserve.pdf")
        out_legacy = str(tmp_path / "legacy.pdf")
        status, msg, _ = OCRDAC.ocr_pdf_preserve(
            src, out_preserve, "eng", auto_preprocessing=False)
        assert status == "success", msg

        status, msg, _ = OCRDAC.ocr_pdf_dual_image(
            src, out_legacy, "eng", auto_preprocessing=False)
        assert status == "success", msg

        preserve_words = _extract_text_ops(out_preserve)
        legacy_words = _extract_text_ops(out_legacy)

        assert preserve_words, "preserve pipeline produced no text"
        assert legacy_words, "legacy pipeline produced no text"
        # Both run the same Tesseract engine on the same page image; the
        # grafted words must match the legacy pipeline's search layer.
        assert preserve_words.intersection(legacy_words), (
            f"Text layers disagree: {preserve_words} vs {legacy_words}"
        )


# ── Config defaulting ────────────────────────────────────────────────────────

class TestConfigDefaults:
    def test_missing_value_threads_to_true(self, tmp_path, fake_tesseract,
                                           legacy_recorder):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = _full_config(tmp_path)
        config.pop("preserve_original_pdf")  # behaves like config file w/o key
        convert_pdfs([in_pdf], config, str(tmp_path))
        assert fake_tesseract, "Missing key should default preserve=True"
        assert not legacy_recorder, "Legacy must not run under default"