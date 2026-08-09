"""Tests for the invisible_text_layer feature (PDF rendering mode 3).

Verifies:
  1. With invisible_text_layer=true (default), grafted OCR text uses PDF
     rendering mode 3 (invisible): the words remain in the content stream
     (searchable/selectable/copyable) but are not painted.
  2. With invisible_text_layer=false, the grafted text uses rendering
     mode 0 (default, visible) for debugging.
  3. The original PDF image layer is untouched (no rasterization, no
     image replacement, same XObjects/bytes).
  4. Magazine-clipping style embedded JPEGs stay byte-identical.
  5. OCR accuracy is unchanged: same words grafted regardless of the flag
     (same HOCR input, same Tesseract engine).
Also verifies config defaulting (missing key => true) and threading
through convert_pdfs.
"""

import io
import os
import re
import sys
import subprocess

import pytest
import pikepdf
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import OCRDAC

GS_AVAILABLE = any(
    os.access(os.path.join(p, "gs"), os.X_OK)
    for p in os.environ["PATH"].split(os.pathsep)
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_small_pdf(path, size=(600, 300)):
    """Single-page image-only PDF (no text layer)."""
    img = Image.new("RGB", size, (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.text((40, 40), "FORM 2026", fill=(0, 0, 0))
    img.save(path, dpi=(300, 300))


def _grafted_streams(pdf_path):
    """Return raw bytes of every content stream containing a BT/ET block."""
    streams = []
    with pikepdf.Pdf.open(pdf_path) as pdf:
        for page in pdf.pages:
            entries = page.obj.get("/Contents")
            blobs = entries if isinstance(entries, pikepdf.Array) else [entries]
            for s in blobs:
                if s is None:
                    continue
                data = s.read_bytes()
                if b"BT" in data and b"ET" in data:
                    streams.append(data)
    return streams


def _text_ops(pdf_path):
    """All `(...) Tj` payloads (searchable text) across grafted streams."""
    texts = []
    for data in _grafted_streams(pdf_path):
        texts.extend(re.findall(rb"\((.*?)\)\s*Tj", data))
    return {t.decode("latin-1") for t in texts if t}


def _uses_rendering_mode(pdf_path, mode):
    """True if any grafted stream sets text rendering mode `mode`."""
    for data in _grafted_streams(pdf_path):
        for m in re.finditer(rb"(\d+)\s+Tr", data):
            if int(m.group(1)) == mode:
                return True
    return False


def _images_byte_signature(pdf_path):
    """Return {name: raw_bytes} of every image XObject on page 1."""
    with pikepdf.Pdf.open(pdf_path) as pdf:
        page = pdf.pages[0]
        return {str(k): v.read_raw_bytes()
                for k, v in page.images.items()}


def _render_to_png(pdf_path, dest):
    """Render a PDF page to a PNG file via Ghostscript."""
    result = subprocess.run(
        ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", "-r150",
         f"-sOutputFile={dest}", pdf_path],
        capture_output=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr[:500]
    with open(dest, "rb") as f:
        return f.read()


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


# ── Test 1: Invisible text (default) ─────────────────────────────────────────

class TestInvisibleText:
    def test_rendering_mode_3_and_searchable(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)

        status, msg, _ = OCRDAC.ocr_pdf_preserve(
            src, out, "eng", auto_preprocessing=False)
        assert status == "success", f"Pipeline failed: {msg}"

        assert _uses_rendering_mode(out, 3), (
            "Grafted text must set PDF rendering mode 3 (invisible)"
        )
        words = _text_ops(out)
        assert "FORM" in words, f"Grafted words missing: {words}"
        assert "2026" in words

    def test_default_config_is_true(self, tmp_path):
        import configparser
        ini = tmp_path / "cfg.ini"
        ini.write_text("[OCRDAC]\nignore = 1\n")
        assert OCRDAC.load_config(str(ini))["invisible_text_layer"] == "true"

    @pytest.mark.skipif(not GS_AVAILABLE, reason="ghostscript not installed")
    def test_invisible_layer_does_not_paint(self, tmp_path, fake_tesseract):
        """Rendered output must be pixel-identical to the input (mode 3
        paints nothing, and the original image layer is untouched)."""
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)

        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        assert _render_to_png(src, str(tmp_path / "src.png")) == \
            _render_to_png(out, str(tmp_path / "out.png")), (
            "Invisible text layer changed the rendered page"
        )

    def test_grafted_stream_first_op_after_bt_is_3_tr(
        self, tmp_path, fake_tesseract
    ):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        for data in _grafted_streams(out):
            bt = data.find(b"BT")
            assert bt != -1
            # Right after BT (or a newline) the rendering mode must be set.
            rest = data[bt + 2:].lstrip(b"\r\n")
            assert rest.startswith(b"3 Tr"), f"Stream does not open with 3 Tr: {rest[:40]!r}"

    def test_no__Tr_revert_before_ET(self, tmp_path, fake_tesseract):
        """Mode 3 must remain in effect for the whole text object; no
        stray '0 Tr' inside the BT..ET block."""
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        for data in _grafted_streams(out):
            bt = data.find(b"BT")
            et = data.find(b"ET", bt)
            block = data[bt:et]
            assert not re.search(rb"\b0\s+Tr\b", block), "rendering mode reset"

    def test_log_line_invisible(self, tmp_path, fake_tesseract, capsys):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)
        assert "OCR text layer: invisible (rendering_mode=3)" in capsys.readouterr().out


# ── Test 2: Visible text (debug mode) ────────────────────────────────────────

class TestVisibleText:
    def test_rendering_mode_0_without_flag(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)

        status, msg, _ = OCRDAC.ocr_pdf_preserve(
            src, out, "eng", auto_preprocessing=False,
            invisible_text_layer=False)
        assert status == "success", f"Pipeline failed: {msg}"

        assert not _uses_rendering_mode(out, 3), (
            "invisible_text_layer=false must NOT set rendering mode 3"
        )
        words = _text_ops(out)
        assert "FORM" in words, f"Grafted words missing: {words}"

    def test_logging_visible(self, tmp_path, fake_tesseract, capsys):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        OCRDAC.ocr_pdf_preserve(
            src, out, "eng", auto_preprocessing=False,
            invisible_text_layer=False)
        assert "OCR text layer: visible (rendering_mode=0)" in capsys.readouterr().out


# ── Test 3: Original PDF preserved ───────────────────────────────────────────

class TestOriginalPreserved:
    def test_images_and_mediabox_unchanged(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)

        before = _images_byte_signature(src)
        with pikepdf.Pdf.open(src) as pdf:
            media_before = [float(x) for x in pdf.pages[0].MediaBox]
            width_before = pdf.pages[0].images.get("/image").get(pikepdf.Name.Width)

        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)

        after = _images_byte_signature(out)
        assert list(after.keys()) == list(before.keys()), "XObject set changed"
        for name in before:
            assert after[name] == before[name], f"Image {name} was altered"
        with pikepdf.Pdf.open(out) as pdf:
            assert [float(x) for x in pdf.pages[0].MediaBox] == media_before
            assert pdf.pages[0].images.get("/image").get(pikepdf.Name.Width) == width_before

    def test_no_new_images(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "in.pdf")
        out = str(tmp_path / "out.pdf")
        _make_small_pdf(src)
        n_before = len(_images_byte_signature(src))
        OCRDAC.ocr_pdf_preserve(src, out, "eng", auto_preprocessing=False)
        assert len(_images_byte_signature(out)) == n_before, (
            "Invisible text grafting must not add/replace image XObjects"
        )


# ── Test 4: Magazine clipping untouched ──────────────────────────────────────

class TestMagazineClipping:
    def test_embedded_jpeg_bytes_identical(self, tmp_path, fake_tesseract):
        src = str(tmp_path / "clip.pdf")
        out = str(tmp_path / "out.pdf")

        # Simulate a chart/photo page ("magazine clipping") stored as a
        # JPEG-encoded PDF page, as a scanner app would embed it.
        chart = Image.new("RGB", (800, 600), (240, 240, 240))
        d = ImageDraw.Draw(chart)
        for x in range(50, 750, 60):
            d.line([(x, 300), (x + 40, 100 + (x % 7) * 20)],
                   fill=(200, 40, 40), width=6)
        buf = io.BytesIO()
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


# ── Test 5: OCR accuracy unchanged ───────────────────────────────────────────

class TestOcrAccuracyUnchanged:
    def test_same_words_in_both_modes(self, tmp_path, fake_tesseract):
        """Same HOCR input must produce the identical word set with or
        without the invisible flag (same Tesseract engine, same boxes)."""
        src = str(tmp_path / "form.pdf")
        _make_small_pdf(src)
        out_hidden = str(tmp_path / "hidden.pdf")
        out_visible = str(tmp_path / "visible.pdf")

        OCRDAC.ocr_pdf_preserve(
            src, out_hidden, "eng", auto_preprocessing=False, invisible_text_layer=True)
        OCRDAC.ocr_pdf_preserve(
            src, out_visible, "eng", auto_preprocessing=False, invisible_text_layer=False)

        assert _text_ops(out_hidden) == _text_ops(out_visible), (
            "Rendering mode must not change the OCR text layer content"
        )
        assert "FORM" in _text_ops(out_hidden) and "2026" in _text_ops(out_hidden)


# ── Config threading ─────────────────────────────────────────────────────────

class TestConfigThreading:
    def _make(self, tmp_path, invisible):
        return {
            "output_dir": None,
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
            "invisible_text_layer": str(invisible).lower(),
        }

    def test_false_threads_through_convert(self, tmp_path, fake_tesseract):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = self._make(tmp_path, invisible=False)
        config["output_dir"] = str(tmp_path / "out")
        config["directory"] = str(tmp_path)
        OCRDAC.convert_pdfs([in_pdf], config, str(tmp_path))
        assert not _uses_rendering_mode(str(tmp_path / "out" / "in.pdf"), 3)

    def test_true_threads_through_convert(self, tmp_path, fake_tesseract):
        in_pdf = str(tmp_path / "in.pdf")
        _make_small_pdf(in_pdf)
        config = self._make(tmp_path, invisible=True)
        config["output_dir"] = str(tmp_path / "out")
        config["directory"] = str(tmp_path)
        OCRDAC.convert_pdfs([in_pdf], config, str(tmp_path))
        assert _uses_rendering_mode(str(tmp_path / "out" / "in.pdf"), 3)