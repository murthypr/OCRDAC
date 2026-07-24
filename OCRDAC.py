#!/usr/bin/env python3
"""
OCRDAC - OCR Document Analysis & Conversion (v2 Dual-Image Pipeline)

Scans directories for PDFs, detects which need OCR, and converts them
using a dual-image pipeline that preserves the original visual layer
pixel-for-pixel while applying preprocessing (median filter + threshold)
ONLY to the OCR input image.

Pipeline per page:
  1. Extract original page image (image_original) — never modified.
  2. Create OCR-only copy (image_for_ocr).
  3. If auto_preprocessing detects low contrast / stripes / uneven
     background, apply median filter + threshold to image_for_ocr.
  4. Feed ONLY image_for_ocr to OCRmyPDF with flags:
       --force-ocr --skip-text --image-dpi 300
  5. Replace the image layer in the OCRmyPDF output with the original
     image_original, preserving crisp visuals + full OCR text.
  6. Write provenance metadata.
"""

import subprocess
import os
import sys
import time
import shutil
import configparser
import io
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
    "ocr_output_file": "ocr_files.txt",
    "non_ocr_output_file": "non_ocr_files.txt",
    "progress_interval": "25",
    "overwrite": "false",
    "skip_existing_ocr": "true",
    "preprocessing": "none",
    "median_filter_size": "3",
    "threshold": "130",
    "auto_preprocessing": "true",
    "ocrdac_version": "v0.4",
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
    ocrdac_version="v0.4"
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
        ocr_images = []
        page_reasons = []
        any_preprocessed = False
        last_reason = "none"
        clean_pages = 0
        preprocessed_pages = 0
        preprocessing_reasons = {}

        for i, img in enumerate(original_images):
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
            page_reasons.append(reason)

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
            ocr_images_pdf,
            ocr_pdf_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
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

        stats = {
            "clean_pages": clean_pages,
            "preprocessed_pages": preprocessed_pages,
            "preprocessing_reasons": preprocessing_reasons,
        }
        return "success", "", stats


# ── Scanning ─────────────────────────────────────────────────────────────────

def scan_directory(root_dir, config):
    """Scan directory for PDFs and categorize them."""
    root_dir = os.path.abspath(root_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_output_path = os.path.join(script_dir, config["ocr_output_file"])
    non_ocr_output_path = os.path.join(script_dir, config["non_ocr_output_file"])
    progress_interval = int(config["progress_interval"])

    results = {
        "ocr": [],
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
                    else:
                        results["ocr"].append(fullpath)
                        ocr_file.write(fullpath + "\n")
                        ocr_file.flush()

                    print(".", end="", flush=True)
                    if results["processed"] % progress_interval == 0:
                        print(f" Processed {results['processed']} files. "
                              f"(ocr:{len(results['ocr'])} / non-ocr:{len(results['non_ocr'])})")

    return results


# ── Copy Already-OCR'd PDFs ─────────────────────────────────────────────────

def copy_ocr_pdfs(ocr_files, config, script_dir):
    """Copy already-OCR'd PDFs to the output directory."""
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
    """Convert non-OCR PDFs using the dual-image pipeline (prod mode)."""
    output_dir = config["output_dir"]
    languages = config["ocr_languages"]
    overwrite = config["overwrite"].lower() == "true"
    skip_existing = config["skip_existing_ocr"].lower() == "true"
    preprocessing = config["preprocessing"].lower()
    filter_size = int(config["median_filter_size"])
    threshold_val = int(config["threshold"])
    ocrdac_version = config["ocrdac_version"]
    auto_preprocessing = config["auto_preprocessing"].lower() == "true"

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

    if auto_preprocessing:
        print("Dual-image pipeline: ENABLED (original visuals preserved, auto-preprocessing for OCR)")
    else:
        print(f"Dual-image pipeline: ENABLED (original visuals preserved, preprocessing={preprocessing})")

    for i, input_path in enumerate(non_ocr_files, 1):
        rel_path = os.path.relpath(input_path, config["directory"])

        if output_dir:
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = input_path

        print(f"[{i}/{len(non_ocr_files)}] {rel_path} ... \n", end="", flush=True)
        status, msg, stats = ocr_pdf_dual_image(
            input_path, output_path, languages,
            auto_preprocessing=auto_preprocessing,
            preprocessing_setting=preprocessing,
            median_filter_size=filter_size,
            threshold_val=threshold_val,
            ocrdac_version=ocrdac_version,
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


def print_summary(results, convert_results, elapsed_seconds):
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
    if mode == "prod":
        if results["ocr"]:
            copy_ocr_pdfs(results["ocr"], config, script_dir)
        if results["non_ocr"]:
            convert_results = convert_pdfs(results["non_ocr"], config, script_dir)
        elif not results["ocr"]:
            print("\nNo PDFs need OCR!")

    overall_elapsed = int(time.monotonic() - overall_start)
    print_summary(results, convert_results, overall_elapsed)
