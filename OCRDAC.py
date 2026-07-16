#!/usr/bin/env python3
"""
OCRDAC - OCR Document Analysis & Conversion
Scans directories for PDFs, detects which need OCR, and optionally converts them.
"""

import subprocess
import os
import sys
import time
import configparser
import shutil
from datetime import datetime, timezone
from pathlib import Path

from preprocessing_detector import detect_preprocessing_needed

CONFIG_FILE = "ocrdac.config"
DEFAULT_CONFIG = {
    "mode": "dry-run",           # "dry-run" or "prod"
    "directory": "./pdfs",
    "output_dir": "",            # empty = overwrite in place
    "ocr_languages": "eng",
    "ocrmypdf_options": "--deskew --clean",
    "ocr_output_file": "ocr_files.txt",
    "non_ocr_output_file": "non_ocr_files.txt",
    "progress_interval": "25",
    "overwrite": "false",
    "skip_existing_ocr": "true",
    "preprocessing": "none",       # "none" or "median"
    "median_filter_size": "3",
    "threshold": "130",
    "auto_preprocessing": "true",  # auto-detect per-page preprocessing needs
    "ocrdac_version": "v0.2",
}

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

# Fix pdf_needs_ocr: use pikepdf instead of missing --is-text-visible flag
# ocrmypdf --is-text-visible does not exist in v16.13.0. The command
# was failing with exit code 2, and the code treated any non-6 return
# as "has text layer", so every PDF was falsely classified as already
# OCR'd.
#
# Replaced the CLI call with pikepdf (already installed as an ocrmypdf
# dependency) to directly inspect page content streams for BT/ET text
# operators. Also added generated files to .gitignore.
def pdf_needs_ocr(path):
    """Returns True if PDF has NO text layer (needs OCR)."""
    import pikepdf
    try:
        pdf = pikepdf.open(path)
    except Exception:
        return True  # Can't open = likely needs OCR or is corrupt
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

def _get_ocrmypdf_version():
    """Return OCRmyPDF version string."""
    try:
        result = subprocess.run(["ocrmypdf", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return "unknown"

def _get_ghostscript_version():
    """Return Ghostscript version string."""
    try:
        result = subprocess.run(["gs", "--version"],
                                capture_output=True, text=True, timeout=10)
        return result.stdout.strip()
    except Exception:
        return "unknown"

def write_metadata(output_pdf_path, metadata_dict):
    """Write provenance metadata into a PDF using pikepdf."""
    import pikepdf
    with pikepdf.Pdf.open(output_pdf_path, allow_overwriting_input=True) as pdf:
        for key, value in metadata_dict.items():
            pdf.docinfo["/" + key] = value
        pdf.save(output_pdf_path)

def read_metadata(pdf_path):
    """Read metadata dictionary from a PDF using pikepdf."""
    import pikepdf
    metadata = {}
    with pikepdf.Pdf.open(pdf_path) as pdf:
        if pdf.docinfo:
            for key, value in pdf.docinfo.items():
                clean_key = str(key).lstrip("/")
                metadata[clean_key] = str(value)
    return metadata

def preprocess_page(page_image, filter_size, threshold):
    """Apply median filter + threshold to remove scan artifacts like horizontal stripes."""
    from PIL import Image, ImageFilter
    gray = page_image.convert("L")
    median = gray.filter(ImageFilter.MedianFilter(filter_size))
    return median.point(lambda x: 255 if x > threshold else 0)

def render_pages_to_images(pdf_path, dpi=300):
    """Render each PDF page to a PIL Image using ghostscript."""
    from PIL import Image
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

def _attach_metadata(output_path, ocrdac_version, ocrmypdf_options, preprocessing,
                     auto_preprocessing_enabled=False, auto_preprocessing_reason="none"):
    """Collect provenance metadata and write it into the output PDF."""
    ocrmypdf_ver = _get_ocrmypdf_version()
    gs_ver = _get_ghostscript_version()
    used_unpaper = "--unpaper" in (ocrmypdf_options or "")
    dpi_norm = "default"

    metadata = {
        "OCRDAC-Version": ocrdac_version,
        "OCRmyPDF-Version": ocrmypdf_ver,
        "Ghostscript-Version": gs_ver,
        "OCR-Flags": ocrmypdf_options or "",
        "OCR-DateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Used-Unpaper": str(used_unpaper),
        "DPI-Normalization": dpi_norm,
        "Auto-Preprocessing-Enabled": str(auto_preprocessing_enabled),
        "Auto-Preprocessing-Reason": auto_preprocessing_reason,
    }

    write_metadata(output_path, metadata)
    print(f"Metadata written: OCRDAC-Version={ocrdac_version}, "
          f"OCRmyPDF={ocrmypdf_ver}, GS={gs_ver}")

def ocr_pdf(input_path, output_path, languages, ocrmypdf_options, overwrite, skip_existing_ocr,
            preprocessing="none", filter_size=3, threshold=130, ocrdac_version="v0.1",
            auto_preprocessing=False):
    """Run OCRmyPDF on a single PDF."""
    if skip_existing_ocr and not pdf_needs_ocr(input_path):
        return "skipped", "Already has text layer"

    if not overwrite and os.path.exists(output_path) and input_path != output_path:
        return "skipped", f"Output exists: {output_path}"

    if preprocessing == "median":
        return _ocr_pdf_preprocessed(input_path, output_path, languages, ocrmypdf_options,
                                     filter_size, threshold, ocrdac_version, auto_preprocessing)

    cmd = ["ocrmypdf"]
    if ocrmypdf_options:
        cmd.extend(ocrmypdf_options.split())
    cmd.extend(["-l", languages, input_path, output_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        _attach_metadata(output_path, ocrdac_version, ocrmypdf_options, preprocessing)
        return "success", ""
    elif result.returncode == 6:
        return "skipped", "Already has text layer"
    else:
        return "error", result.stderr.strip()

def _ocr_pdf_preprocessed(input_path, output_path, languages, ocrmypdf_options,
                           filter_size, threshold, ocrdac_version, auto_preprocessing=False):
    """OCR by preprocessing pages with median filter, then running tesseract + gs.

    When auto_preprocessing is True, each page is analyzed and median filtering
    is applied only to pages that need it. Clean pages are OCR'd directly.
    """
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        images = render_pages_to_images(input_path)
        if not images:
            return "error", "Failed to render pages"

        txt_files = []
        any_preprocessed = False
        last_reason = "none"

        for i, img in enumerate(images):
            page_needs_preprocessing = False
            reason = "none"

            if auto_preprocessing:
                page_needs_preprocessing, reason = detect_preprocessing_needed(img)
                if page_needs_preprocessing:
                    any_preprocessed = True
                    last_reason = reason
                    print(f"  Auto-preprocessing: enabled (reason={reason})", flush=True)
                else:
                    print(f"  Auto-preprocessing: disabled (clean scan)", flush=True)
            else:
                page_needs_preprocessing = True
                reason = "manual"
                any_preprocessed = True
                last_reason = "manual"

            if page_needs_preprocessing:
                processed = preprocess_page(img, filter_size, threshold)
                page_path = os.path.join(tmpdir, f"page_{i:03d}")
                processed.save(f"{page_path}.png")
            else:
                # Save the original page for tesseract
                page_path = os.path.join(tmpdir, f"page_{i:03d}")
                img.save(f"{page_path}.png")

            result = subprocess.run(
                ["tesseract", f"{page_path}.png", page_path,
                 "--psm", "6", "-l", languages, "pdf"],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                return "error", f"tesseract failed on page {i+1}: {result.stderr.strip()}"
            txt_files.append(f"{page_path}.pdf")

        # Merge individual page PDFs
        merge_cmd = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=pdfwrite",
                     f"-sOutputFile={output_path}"] + txt_files
        result = subprocess.run(merge_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return "error", f"gs merge failed: {result.stderr.strip()}"

        _attach_metadata(output_path, ocrdac_version, ocrmypdf_options, "median",
                         auto_preprocessing_enabled=auto_preprocessing,
                         auto_preprocessing_reason=last_reason if auto_preprocessing else "none")
        return "success", ""

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

    # Clear output files
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

def convert_pdfs(non_ocr_files, config, script_dir):
    """Convert non-OCR PDFs using OCRmyPDF (prod mode)."""
    output_dir = config["output_dir"]
    languages = config["ocr_languages"]
    ocrmypdf_options = config["ocrmypdf_options"]
    overwrite = config["overwrite"].lower() == "true"
    skip_existing = config["skip_existing_ocr"].lower() == "true"
    preprocessing = config["preprocessing"].lower()
    filter_size = int(config["median_filter_size"])
    threshold = int(config["threshold"])
    ocrdac_version = config["ocrdac_version"]
    auto_preprocessing = config["auto_preprocessing"].lower() == "true"

    results = {"success": 0, "skipped": 0, "errors": []}

    print(f"\n=== PROD MODE: Converting {len(non_ocr_files)} PDFs ===")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print("Mode: Overwrite in place")

    if auto_preprocessing and preprocessing == "median":
        print("Auto-preprocessing: ENABLED (per-page analysis)")
    elif preprocessing == "median":
        print("Auto-preprocessing: DISABLED (apply median to all pages)")
    else:
        print(f"Preprocessing: {preprocessing}")

    for i, input_path in enumerate(non_ocr_files, 1):
        rel_path = os.path.relpath(input_path, config["directory"])

        if output_dir:
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = input_path

        print(f"[{i}/{len(non_ocr_files)}] {rel_path} ... ", end="", flush=True)
        status, msg = ocr_pdf(input_path, output_path, languages, ocrmypdf_options, overwrite, skip_existing,
                              preprocessing, filter_size, threshold, ocrdac_version,
                              auto_preprocessing)

        if status == "success":
            print("✓")
            results["success"] += 1
        elif status == "skipped":
            print(f"⊘ {msg}")
            results["skipped"] += 1
        else:
            print(f"✗ {msg}")
            results["errors"].append((input_path, msg))

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

    print(f"\nResults saved to:")
    print(f"  OCR'd files:     {DEFAULT_CONFIG['ocr_output_file']}")
    print(f"  Non-OCR files:   {DEFAULT_CONFIG['non_ocr_output_file']}")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, CONFIG_FILE)
    config = load_config(config_path)

    # Allow CLI override for directory and mode
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
        # Try relative to script dir
        abs_dir = os.path.join(script_dir, directory)
        if os.path.isdir(abs_dir):
            directory = abs_dir
        else:
            print(f"Directory not found: {directory}")
            sys.exit(1)

    config["directory"] = directory

    print(f"OCRDAC v1.0 - Mode: {mode.upper()}")
    print(f"Scanning: {directory}")
    print("-" * 50)

    start_time = time.perf_counter()
    results = scan_directory(directory, config)
    end_time = time.perf_counter()
    elapsed_seconds = int(end_time - start_time)

    convert_results = None
    if mode == "prod" and results["non_ocr"]:
        convert_results = convert_pdfs(results["non_ocr"], config, script_dir)
    elif mode == "prod":
        print("\nNo PDFs need OCR!")

    print_summary(results, convert_results, elapsed_seconds)
