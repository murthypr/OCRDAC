#!/usr/bin/env python3
"""
OCRDAC - OCR Document Analysis & Conversion

Two OCR pipelines:

1. Preserve-original (default, preserve_original_pdf=true):
   - The original PDF is opened directly with pikepdf and NEVER rasterized.
   - Pages are rasterized (or their embedded images extracted) ONLY to feed
     Tesseract; those images are never embedded in the output.
   - OCR text is grafted onto each original page as a positioned text layer
     (built from HOCR bounding boxes) using pikepdf alone.
   - Ghostscript is never used for the final PDF generation.

2. Legacy dual-image pipeline (preserve_original_pdf=false):
   - Renders pages via Ghostscript, runs OCRmyPDF on preprocessed images,
     then swaps the original visuals back in. Can degrade quality.
"""

import subprocess
import os
import sys
import time
import shutil
import configparser
import shlex
import io
import re
import html as _html
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageFilter
import pikepdf

from preprocessing_detector import detect_preprocessing_needed

CONFIG_FILE = "ocrdac.config"
DEFAULT_CONFIG = {
    "mode": "dry-run",
    "directory": "./pdfs",
    "output_dir": "",
    "ocr_languages": "eng",
    "ocrmypdf_params": "",
    "ocr_output_file": "ocr_files.txt",
    "non_ocr_output_file": "non_ocr_files.txt",
    "progress_interval": "25",
    "preprocessing": "none",
    "median_filter_size": "3",
    "threshold": "130",
    "auto_preprocessing": "true",
    "copy_already_ocred_files": "true",
    "preserve_original_pdf": "true",
    "invisible_text_layer": "true",
    "ocrdac_version": "v0.5",
}


# ── Configuration ────────────────────────────────────────────────────────────

def load_config(config_path):
    """Load configuration from file with defaults."""
    config = DEFAULT_CONFIG.copy()

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        print("Using defaults. Create ocrdac.config to customize.")
        return config

    parser = configparser.ConfigParser()
    parser.read(config_path)

    if "OCRDAC" in parser:
        for key in DEFAULT_CONFIG:
            if key in parser["OCRDAC"]:
                config[key] = parser["OCRDAC"][key].strip().strip('"\'')

    return config


def parse_bool(value, default=False):
    """Parse a boolean config value (case-insensitive).

    Accepts "true"/"false" (also 1/0, yes/no, on/off). Returns default
    when the value is missing or unparsable.
    """
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "on"):
            return True
        if v in ("false", "0", "no", "off"):
            return False
    return default


# ── PDF Inspection ───────────────────────────────────────────────────────────

def pdf_needs_ocr(path):
    """Returns True if PDF has NO text layer (needs OCR)."""
    import pikepdf
    try:
        pdf = pikepdf.open(path)
    except Exception:
        return True
    for page in pdf.pages:
        contents = page.get("/Contents")
        if contents is None:
            continue
        streams = contents if isinstance(contents, pikepdf.Array) else [contents]
        for stream in streams:
            try:
                data = stream.read_bytes()
                if b"BT" in data and b"ET" in data:
                    pdf.close()
                    return False
            except Exception:
                continue
    pdf.close()
    return True


# ── Version Helpers ──────────────────────────────────────────────────────────

def _get_ocrmypdf_version():
    try:
        result = subprocess.run(["ocrmypdf", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return "unknown"


def _get_ghostscript_version():
    try:
        result = subprocess.run(["gs", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ── Metadata Helpers ─────────────────────────────────────────────────────────

def write_metadata(output_pdf_path, metadata_dict):
    """Write provenance metadata into a PDF using pikepdf."""
    with pikepdf.Pdf.open(output_pdf_path, allow_overwriting_input=True) as pdf:
        for key, value in metadata_dict.items():
            pdf.docinfo["/" + key] = value
        pdf.save(output_pdf_path)


def read_metadata(pdf_path):
    """Read metadata dictionary from a PDF using pikepdf."""
    metadata = {}
    with pikepdf.Pdf.open(pdf_path) as pdf:
        if pdf.docinfo:
            for key, value in pdf.docinfo.items():
                clean_key = str(key).lstrip("/")
                metadata[clean_key] = str(value)
    return metadata


# ── Image Helpers ────────────────────────────────────────────────────────────

def render_pages_to_images(pdf_path, dpi=300):
    """Render each PDF page to a PIL Image using ghostscript."""
    import tempfile
    images = []
    with tempfile.TemporaryDirectory() as tmpdir:
        subprocess.run(
            ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", f"-r{dpi}",
             f"-sOutputFile={tmpdir}/page_%03d.png", pdf_path],
            capture_output=True, timeout=300
        )
        for f in sorted(os.listdir(tmpdir)):
            if f.endswith(".png"):
                images.append(Image.open(os.path.join(tmpdir, f)).copy())
    return images


# ── Clean Sandwich Pipeline (preserve_original_pdf) ──────────────────────────

def extract_ocr_images(pdf_path, dpi=300):
    """Return per-page PIL images used ONLY as Tesseract input.

    Strategy (original PDF is never modified):
      1. PyMuPDF: use a page's embedded image when it is the sole, roughly
         page-sized raster (native, lossless extraction).
      2. Otherwise rasterize the page with PyMuPDF at `dpi` dpi.
      3. Fall back to Ghostscript rendering (render_pages_to_images) when
         PyMuPDF is not installed.
    """
    try:
        import pymupdf
    except ImportError:
        return render_pages_to_images(pdf_path, dpi=dpi)

    images = []
    doc = pymupdf.open(pdf_path)
    for page in doc:
        page_rect = page.rect
        used_native = False
        raw = page.get_images(full=True)
        if raw:
            try:
                base = doc.extract_image(raw[0][0])
                data = base["image"]
                pil_img = Image.open(io.BytesIO(data)).convert("RGB")
                aspect_page = page_rect.width / page_rect.height
                aspect_img = pil_img.width / pil_img.height
                if abs(aspect_page - aspect_img) / aspect_page < 0.05:
                    images.append(pil_img)
                    used_native = True
            except Exception:
                pass
        if not used_native:
            pix = page.get_pixmap(dpi=dpi)
            images.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
    doc.close()
    return images


def prepare_ocr_images(images, auto_preprocessing, preprocessing_setting,
                       median_filter_size, threshold_val):
    """Shared preprocessing stage: returns (ocr_images, stats, last_reason).

    Takes the page images destined for OCR input and applies
    auto/manual preprocessing decisions, exactly as the dual-image
    pipeline did, so both pipelines OCR the same images.
    """
    ocr_images = []
    any_preprocessed = False
    last_reason = "none"
    clean_pages = 0
    preprocessed_pages = 0
    preprocessing_reasons = {}

    for i, img in enumerate(images):
        ocr_img = img.copy()
        should_preprocess = False
        reason = "none"

        if auto_preprocessing:
            should_preprocess, reason = detect_preprocessing_needed(img)
        elif preprocessing_setting == "median":
            should_preprocess = True
            reason = "manual"

        if should_preprocess:
            preprocessed_pages += 1
            display_reason = reason
            if display_reason == "stripes":
                display_reason = "stripe_artifacts"
            preprocessing_reasons[display_reason] = preprocessing_reasons.get(display_reason, 0) + 1
            any_preprocessed = True
            last_reason = reason
            ocr_img = preprocess_ocr_image(ocr_img, median_filter_size, threshold_val)
            if auto_preprocessing:
                print(f"  Page {i+1}: Auto-preprocessing: enabled (reason={reason})", flush=True)
            else:
                print(f"  Page {i+1}: Preprocessing: applied (manual)", flush=True)
        else:
            clean_pages += 1
            ocr_img = ocr_img.convert("RGB")
            if auto_preprocessing:
                print(f"  Page {i+1}: Auto-preprocessing: disabled (clean scan)", flush=True)
            else:
                print(f"  Page {i+1}: Preprocessing: not applied", flush=True)

        ocr_images.append(ocr_img)

    stats = {
        "clean_pages": clean_pages,
        "preprocessed_pages": preprocessed_pages,
        "preprocessing_reasons": preprocessing_reasons,
    }
    return ocr_images, stats, last_reason, any_preprocessed


def run_tesseract_hocr(image, tmpdir, index, languages):
    """Run Tesseract on one OCR-only image; returns the HOCR HTML text.

    The image is saved to a temp file purely for Tesseract input; it is
    never embedded in a PDF.
    """
    img_path = os.path.join(tmpdir, f"ocr_input_{index}.png")
    image.save(img_path)
    cmd = ["tesseract", img_path, "stdout", "-l", languages, "--psm", "3", "hocr"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        raise RuntimeError(f"tesseract failed (code {result.returncode}): {result.stderr.strip()}")
    return result.stdout


_HOCR_BBOX_RE = re.compile(
    r"title\s*=\s*[\"']\s*bbox\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)")


def parse_hocr_words(hocr_text):
    """Parse (x0, y0, x1, y1, word) tuples from Tesseract HOCR output.

    Coordinates are in image pixels, origin top-left (HOCR convention).
    HTML entities in the word text are unescaped. Attribute quoting
    (single or double) is handled for both tesseract 4.x and 5.x, and
    word spans nested inside line spans are found by scanning tag text
    rather than assuming a flat structure.
    """
    words = []
    pos = 0
    while True:
        i = hocr_text.find("ocrx_word", pos)
        if i == -1:
            break
        start = hocr_text.rfind("<", 0, i)
        tag_end = hocr_text.find(">", i)
        attrs = hocr_text[start:tag_end]
        bm = _HOCR_BBOX_RE.search(attrs)
        if bm:
            x0, y0, x1, y1 = (float(v) for v in bm.groups())
            close = hocr_text.find("</span>", tag_end)
            body = hocr_text[tag_end + 1:close]
            text = _html.unescape(body).strip()
            if text:
                words.append((x0, y0, x1, y1, text))
        pos = tag_end
    return words


def _pdf_escape(text):
    """Encode unicode OCR text into a PDF string literal body."""
    return text.encode("latin-1", "replace").decode("latin-1")


def graft_hocr_text(pdf, page_adaptive, hocr_text, image_size,
                    invisible_text_layer=True):
    """Graft a positioned OCR text layer onto one PDF page.

    HOCR word boxes (pixels, top-left origin) are mapped into PDF user
    space via the page's MediaBox, so text overlays the original visuals
    without altering any existing content or image XObjects.

    When invisible_text_layer is true the text is rendered with PDF
    rendering mode 3 (invisible): the glyphs are not painted, but the
    text remains fully selectable, copyable, searchable, and indexable.
    Font size/color, opacity, blending mode, and text positioning are
    unchanged; only the rendering mode operator is added.
    """
    words = parse_hocr_words(hocr_text)
    if not words:
        return 0

    img_w, img_h = image_size
    media = page_adaptive.MediaBox  # e.g. [0, 0, w, h] in points
    page_wx = float(media[2])
    page_hy = float(media[3])
    sx = page_wx / img_w
    sy = page_hy / img_h

    ops = ["BT"]
    if invisible_text_layer:
        ops.append("3 Tr")  # rendering mode 3: invisible text
    for x0, y0, x1, y1, text in words:
        x = x0 * sx
        y = page_hy - (y1 * sy)  # flip to PDF bottom-left origin
        size = max(1.0, (y1 - y0) * sy)
        safe = _pdf_escape(text)
        ops.append(f"1 0 0 1 {x:.3f} {y:.3f} Tm /F1 {size:.3f} Tf ({safe}) Tj")
    ops.append("ET")
    body = "\n".join(ops).encode("latin-1")

    # Ensure the page carries an /F1 Helvetica font resource.
    res = page_adaptive.Resources
    if res is None:
        res = pikepdf.Dictionary()
        page_adaptive.Resources = res
    fonts = res.get(pikepdf.Name.Font)
    if fonts is None:
        fonts = pikepdf.Dictionary()
        res[pikepdf.Name.Font] = fonts
    if pikepdf.Name.F1 not in fonts:
        fonts[pikepdf.Name.F1] = pikepdf.Name.Helvetica

    new_stream = pikepdf.Stream(pdf, body)
    contents = page_adaptive.get(pikepdf.Name.Contents)
    if contents is None:
        page_adaptive[pikepdf.Name.Contents] = new_stream
    elif isinstance(contents, pikepdf.Array):
        contents.append(new_stream)
    else:
        page_adaptive[pikepdf.Name.Contents] = pikepdf.Array([contents, new_stream])
    return len(words)


def ocr_pdf_preserve(input_path, output_path, languages,
                     auto_preprocessing=True, preprocessing_setting="none",
                     median_filter_size=3, threshold_val=130,
                     ocrdac_version="v0.5", invisible_text_layer=True):
    """
    Clean-sandwich pipeline: OCR a PDF WITHOUT rasterizing or altering the
    original page content (no Ghostscript at the final stage).

    Returns (status, message) where status is "success", "skipped", or "error".
    """
    EMPTY_STATS = {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}

    if not pdf_needs_ocr(input_path):
        return "skipped", "Already has text layer", EMPTY_STATS

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. OCR INPUT ONLY: rasterize/extract pages for Tesseract ─────────
        ocr_input_images = extract_ocr_images(input_path)
        if not ocr_input_images:
            return "error", "Failed to extract page images for OCR", EMPTY_STATS

        # ── 2. Preprocess OCR input (never embedded in the output) ────────────
        ocr_images, stats, last_reason, any_preprocessed = prepare_ocr_images(
            ocr_input_images, auto_preprocessing, preprocessing_setting,
            median_filter_size, threshold_val)

        # ── 3. Open original untouched; run Tesseract; graft text ─────────────
        overwrite_input = (os.path.abspath(input_path) == os.path.abspath(output_path))
        with pikepdf.Pdf.open(input_path, allow_overwriting_input=overwrite_input) as pdf:
            page_count = len(pdf.pages)
            if len(ocr_images) < page_count:
                return "error", f"OCR input pages ({len(ocr_images)}) < PDF pages ({page_count})", EMPTY_STATS

            for i, page in enumerate(pdf.pages):
                hocr = run_tesseract_hocr(ocr_images[i], tmpdir, i, languages)
                img_size = ocr_images[i].size
                graft_hocr_text(pdf, page, hocr, img_size,
                                invisible_text_layer=invisible_text_layer)
                if invisible_text_layer:
                    print(f"  Page {i+1}: OCR text layer: invisible (rendering_mode=3)",
                          flush=True)
                else:
                    print(f"  Page {i+1}: OCR text layer: visible (rendering_mode=0)",
                          flush=True)

            # ── 4. Provenance metadata ─────────────────────────────────────────
            meta_reason = last_reason if any_preprocessed else "none"
            pdf.docinfo["/OCRDAC-Version"] = ocrdac_version
            pdf.docinfo["/Auto-Preprocessing-Enabled"] = str(auto_preprocessing)
            pdf.docinfo["/Auto-Preprocessing-Reason"] = meta_reason
            pdf.docinfo["/Median-Filter-Used"] = str(any_preprocessed)
            pdf.docinfo["/Median-Filter-Size"] = str(median_filter_size)
            pdf.docinfo["/Threshold-Used"] = str(threshold_val)
            pdf.docinfo["/OCRmyPDF-Version"] = _get_ocrmypdf_version()
            pdf.docinfo["/Ghostscript-Version"] = _get_ghostscript_version()
            pdf.docinfo["/OCR-DateTime"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            pdf.save(output_path)

        return "success", "", stats


# ── Dual‑Image OCR Pipeline ──────────────────────────────────────────────────

def preprocess_ocr_image(image, filter_size, threshold_val):
    """Apply median filter + threshold to prepare an image for OCR."""
    gray = image.convert("L")
    median = gray.filter(ImageFilter.MedianFilter(filter_size))
    binary = median.point(lambda x: 255 if x > threshold_val else 0)
    return binary.convert("RGB")


def ocr_pdf_dual_image(
    input_path, output_path, languages,
    auto_preprocessing=True, preprocessing_setting="none",
    median_filter_size=3, threshold_val=130,
    ocrdac_version="v0.5",
    ocrmypdf_params=""
):
    """
    Dual‑image OCR pipeline for a single PDF.

    Returns (status, message) where status is "success", "skipped", or "error".
    """
    EMPTY_STATS = {"clean_pages": 0, "preprocessed_pages": 0, "preprocessing_reasons": {}}

    if not pdf_needs_ocr(input_path):
        return "skipped", "Already has text layer", EMPTY_STATS

    import tempfile
    import io as _io

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── 1. Render original pages to images ────────────────────────────────
        original_images = render_pages_to_images(input_path)
        if not original_images:
            return "error", "Failed to render pages from PDF", EMPTY_STATS

        # ── 2. Create OCR copies and optionally preprocess ────────────────────
        ocr_images, stats_internal, last_reason, any_preprocessed = prepare_ocr_images(
            original_images, auto_preprocessing, preprocessing_setting,
            median_filter_size, threshold_val)

        # ── 3. Save OCR-only images as a single PDF ──────────────────────────
        #     OCRmyPDF accepts one input (PDF or single image), so we combine
        #     the preprocessed images into a multi-page PDF first.
        #     We set dpi=(300,300) so the PDF page dimensions are correct
        #     (image pixels / 300 * 72 = points).
        ocr_images_pdf = os.path.join(tmpdir, "ocr_images.pdf")
        first_ocr = ocr_images[0]
        first_ocr.save(
            ocr_images_pdf,
            save_all=True,
            append_images=ocr_images[1:] if len(ocr_images) > 1 else [],
            dpi=(300, 300)
        )

        # ── 4. Run OCRmyPDF on the OCR-only PDF ──────────────────────────────
        #     Flags: --force-ocr --image-dpi 300
        #     --skip-text is mutually exclusive with --force-ocr in OCRmyPDF,
        #     so it is omitted. --force-ocr ensures OCR runs on every page.
        #     --clean, --deskew, --remove-background, --rotate-pages,
        #     --optimize are deliberately NOT used (they modify the image).
        ocr_pdf_path = os.path.join(tmpdir, "ocr_result.pdf")
        cmd = [
            "ocrmypdf",
            "--force-ocr",
            "--image-dpi", "300",
            "-l", languages,
        ]
        # Insert user-controlled OCRmyPDF flags (from ocrmypdf_params in
        # ocrdac.config) BEFORE the input/output positional paths. OCRmyPDF
        # rejects flags placed after the two positional arguments, and
        # argparse would otherwise try to consume the input path as a flag
        # value (e.g. --verbose <input.pdf>). User flags keep their order and
        # OCRDAC's built-in flags are preserved.
        if ocrmypdf_params:
            cmd.extend(shlex.split(ocrmypdf_params))
        cmd.extend([ocr_images_pdf, ocr_pdf_path])

        verbose_mode = "--verbose" in cmd
        if verbose_mode:
            print("  OCRmyPDF command:", " ".join(cmd), flush=True)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if verbose_mode:
            if result.stdout:
                print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
            if result.stderr:
                sys.stderr.write(result.stderr)
                if not result.stderr.endswith("\n"):
                    sys.stderr.write("\n")
        if result.returncode not in (0, 6):
            return "error", f"OCRmyPDF failed (code {result.returncode}): {result.stderr.strip()}", EMPTY_STATS

        # ── 5. Replace OCRmyPDF's image layer with original images ────────────
        with pikepdf.Pdf.open(ocr_pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                orig = original_images[i]
                if orig.mode != "RGB":
                    orig = orig.convert("RGB")

                # Raw RGB pixel data compressed with zlib (FlateDecode)
                raw_bytes = orig.tobytes()
                import zlib as _zlib
                compressed = _zlib.compress(raw_bytes)

                img_names = list(page.images.keys())
                for name in img_names:
                    new_stream = pikepdf.Stream(pdf, compressed)
                    new_stream.Type = pikepdf.Name.XObject
                    new_stream.Subtype = pikepdf.Name.Image
                    new_stream.Width = orig.width
                    new_stream.Height = orig.height
                    new_stream.ColorSpace = pikepdf.Name.DeviceRGB
                    new_stream.BitsPerComponent = 8
                    new_stream.Filter = pikepdf.Name.FlateDecode

                    res = page.Resources
                    if res is not None:
                        xobj = res.XObject
                        if xobj is not None:
                            xobj[name] = new_stream

            # ── 6. Write provenance metadata ──────────────────────────────────
            meta_reason = last_reason if any_preprocessed else "none"
            pdf.docinfo["/OCRDAC-Version"] = ocrdac_version
            pdf.docinfo["/Auto-Preprocessing-Enabled"] = str(auto_preprocessing)
            pdf.docinfo["/Auto-Preprocessing-Reason"] = meta_reason
            pdf.docinfo["/Median-Filter-Used"] = str(any_preprocessed)
            pdf.docinfo["/Median-Filter-Size"] = str(median_filter_size)
            pdf.docinfo["/Threshold-Used"] = str(threshold_val)
            pdf.docinfo["/OCRmyPDF-Version"] = _get_ocrmypdf_version()
            pdf.docinfo["/Ghostscript-Version"] = _get_ghostscript_version()
            pdf.docinfo["/OCR-DateTime"] = (
                datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            )

            pdf.save(output_path)

        return "success", "", stats_internal


# ── Scanning ─────────────────────────────────────────────────────────────────

def scan_directory(root_dir, config):
    """Scan directory for PDFs and categorize them."""
    root_dir = os.path.abspath(root_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_output_path = os.path.join(script_dir, config["ocr_output_file"])
    non_ocr_output_path = os.path.join(script_dir, config["non_ocr_output_file"])
    progress_interval = int(config["progress_interval"])
    copy_already_ocred = config.get("copy_already_ocred_files", "true").lower() == "true"

    results = {
        "ocr": [],
        "ocr_count": 0,
        "non_ocr": [],
        "processed": 0,
        "errors": []
    }

    open(ocr_output_path, "w").close()
    open(non_ocr_output_path, "w").close()

    with open(ocr_output_path, "a", encoding="utf-8") as ocr_file, \
         open(non_ocr_output_path, "a", encoding="utf-8") as non_ocr_file:

        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(".pdf"):
                    fullpath = os.path.join(dirpath, name)

                    needs_ocr = pdf_needs_ocr(fullpath)
                    results["processed"] += 1

                    if needs_ocr:
                        results["non_ocr"].append(fullpath)
                        non_ocr_file.write(fullpath + "\n")
                        non_ocr_file.flush()
                    elif copy_already_ocred:
                        results["ocr"].append(fullpath)
                        results["ocr_count"] += 1
                        ocr_file.write(fullpath + "\n")
                        ocr_file.flush()

                    print(".", end="", flush=True)
                    if results["processed"] % progress_interval == 0:
                        print(f" Processed {results['processed']} files. "
                              f"(ocr:{len(results['ocr'])} / non-ocr:{len(results['non_ocr'])})")

    return results


# ── Copy Already-OCR'd PDFs ─────────────────────────────────────────────────

def copy_ocr_pdfs(ocr_files, config, script_dir):
    """Copy already-OCR'd PDFs to the output directory.

    Controlled by config key copy_already_ocred_files. When false, the
    already-OCR'd PDFs are skipped entirely (no copying).
    """
    if config.get("copy_already_ocred_files", "true").lower() != "true":
        return 0

    output_dir = config["output_dir"]
    if not output_dir:
        return 0

    print(f"\n=== PROD MODE: Copying {len(ocr_files)} already-OCR'd PDFs ===")
    if output_dir:
        print(f"Output directory: {output_dir}")

    copied = 0
    for filepath in ocr_files:
        rel_path = os.path.relpath(filepath, config["directory"])
        dest = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(filepath, dest)
        copied += 1

    print(f"  Copied {copied} file(s).")
    return copied


# ── Conversion Orchestration ─────────────────────────────────────────────────

def convert_pdfs(non_ocr_files, config, script_dir):
    """Convert non-OCR PDFs (prod mode).

    Uses the clean-sandwich pipeline when preserve_original_pdf=true
    (default) and the legacy Ghostscript dual-image pipeline otherwise.
    """
    output_dir = config["output_dir"]
    languages = config["ocr_languages"]
    preprocessing = config["preprocessing"].lower()
    filter_size = int(config["median_filter_size"])
    threshold_val = int(config["threshold"])
    ocrdac_version = config["ocrdac_version"]
    auto_preprocessing = config["auto_preprocessing"].lower() == "true"
    preserve_original = parse_bool(
        config.get("preserve_original_pdf", ""), default=True)
    invisible_text_layer = parse_bool(
        config.get("invisible_text_layer", ""), default=True)

    results = {"success": 0, "skipped": 0, "errors": []}
    agg_clean = 0
    agg_preprocessed = 0
    agg_reasons = {}

    print(f"\n=== PROD MODE: Converting {len(non_ocr_files)} PDFs ===")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print("Mode: Overwrite in place")

    if preserve_original:
        print("Clean-sandwich pipeline: ENABLED (original PDF preserved, "
              "OCR text grafted via pikepdf, no Ghostscript)")
    elif auto_preprocessing:
        print("Dual-image pipeline: ENABLED (original visuals preserved, auto-preprocessing for OCR)")
    else:
        print(f"Dual-image pipeline: ENABLED (original visuals preserved, preprocessing={preprocessing})")
    if invisible_text_layer:
        print("OCR text layer: invisible (rendering_mode=3)")
    else:
        print("OCR text layer: visible (rendering_mode=0)")

    for i, input_path in enumerate(non_ocr_files, 1):
        rel_path = os.path.relpath(input_path, config["directory"])

        if output_dir:
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = input_path

        print(f"[{i}/{len(non_ocr_files)}] {rel_path} ... \n", end="", flush=True)
        if preserve_original:
            status, msg, stats = ocr_pdf_preserve(
                input_path, output_path, languages,
                auto_preprocessing=auto_preprocessing,
                preprocessing_setting=preprocessing,
                median_filter_size=filter_size,
                threshold_val=threshold_val,
                ocrdac_version=ocrdac_version,
                invisible_text_layer=invisible_text_layer,
            )
        else:
            status, msg, stats = ocr_pdf_dual_image(
                input_path, output_path, languages,
                auto_preprocessing=auto_preprocessing,
                preprocessing_setting=preprocessing,
                median_filter_size=filter_size,
                threshold_val=threshold_val,
                ocrdac_version=ocrdac_version,
                ocrmypdf_params=config.get("ocrmypdf_params", ""),
            )

        if status == "success":
            agg_clean += stats["clean_pages"]
            agg_preprocessed += stats["preprocessed_pages"]
            for reason, count in stats["preprocessing_reasons"].items():
                agg_reasons[reason] = agg_reasons.get(reason, 0) + count
            print("\u2713")
            results["success"] += 1
        elif status == "skipped":
            print(f"\u2298 {msg}")
            results["skipped"] += 1
        else:
            print(f"\u2717 {msg}")
            results["errors"].append((input_path, msg))

    results["preprocessing_stats"] = {
        "clean_pages": agg_clean,
        "preprocessed_pages": agg_preprocessed,
        "preprocessing_reasons": agg_reasons,
    }
    return results


def print_summary(results, convert_results, elapsed_seconds, ocr_copied=None):
    """Print summary of results."""
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60

    print("\n" + "=" * 50)
    print("OCRDAC SUMMARY")
    print("=" * 50)
    print(f"Time elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Total processing time: {elapsed_seconds}s")
    print(f"Files scanned: {results['processed']}")
    print(f"  Already OCR'd: {len(results['ocr'])}")
    print(f"  Need OCR:      {len(results['non_ocr'])}")

    if ocr_copied:
        print(f"  Copied (already OCR'd): {ocr_copied}")

    if convert_results:
        print(f"\nConversion results:")
        print(f"  Successful: {convert_results['success']}")
        print(f"  Skipped:    {convert_results['skipped']}")
        print(f"  Errors:     {len(convert_results['errors'])}")
        if convert_results["errors"]:
            print("\nErrors:")
            for path, err in convert_results["errors"]:
                print(f"  {path}: {err}")

    if convert_results and convert_results.get("preprocessing_stats"):
        stats = convert_results["preprocessing_stats"]
        if stats["clean_pages"] > 0 or stats["preprocessed_pages"] > 0:
            print(f"\nAuto-Preprocessing Summary:")
            print(f"  Clean pages (no preprocessing needed): {stats['clean_pages']}")
            print(f"  Preprocessed pages: {stats['preprocessed_pages']}")
            if stats["preprocessing_reasons"]:
                print(f"  Reasons:")
                for reason, count in sorted(stats["preprocessing_reasons"].items()):
                    print(f"    - {reason}: {count}")

    print(f"\nResults saved to:")
    print(f"  OCR'd files:     {DEFAULT_CONFIG['ocr_output_file']}")
    print(f"  Non-OCR files:   {DEFAULT_CONFIG['non_ocr_output_file']}")


# ── Main Entry Point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, CONFIG_FILE)
    config = load_config(config_path)

    if len(sys.argv) > 1:
        config["directory"] = sys.argv[1]
    if len(sys.argv) > 2:
        config["mode"] = sys.argv[2]

    mode = config["mode"].lower()
    if mode not in ("dry-run", "prod"):
        print(f"Invalid mode: {mode}. Use 'dry-run' or 'prod'")
        sys.exit(1)

    directory = config["directory"]
    if not os.path.isdir(directory):
        abs_dir = os.path.join(script_dir, directory)
        if os.path.isdir(abs_dir):
            directory = abs_dir
        else:
            print(f"Directory not found: {directory}")
            sys.exit(1)

    config["directory"] = directory

    print(f"OCRDAC v2.0 - Mode: {mode.upper()}")
    print(f"Scanning: {directory}")
    print("-" * 50)

    overall_start = time.monotonic()
    results = scan_directory(directory, config)

    convert_results = None
    ocr_copied = 0
    if mode == "prod":
        if results["ocr"]:
            ocr_copied = copy_ocr_pdfs(results["ocr"], config, script_dir)
        if results["non_ocr"]:
            convert_results = convert_pdfs(results["non_ocr"], config, script_dir)
        elif not results["ocr"]:
            print("\nNo PDFs need OCR!")

    overall_elapsed = int(time.monotonic() - overall_start)
    print_summary(results, convert_results, overall_elapsed, ocr_copied=ocr_copied)
