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
from pathlib import Path

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

def ocr_pdf(input_path, output_path, languages, ocrmypdf_options, overwrite, skip_existing_ocr):
    """Run OCRmyPDF on a single PDF."""
    if skip_existing_ocr and not pdf_needs_ocr(input_path):
        return "skipped", "Already has text layer"

    if not overwrite and os.path.exists(output_path) and input_path != output_path:
        return "skipped", f"Output exists: {output_path}"

    cmd = ["ocrmypdf"]
    if ocrmypdf_options:
        cmd.extend(ocrmypdf_options.split())
    cmd.extend(["-l", languages, input_path, output_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return "success", ""
    elif result.returncode == 6:
        return "skipped", "Already has text layer"
    else:
        return "error", result.stderr.strip()

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

    results = {"success": 0, "skipped": 0, "errors": []}

    print(f"\n=== PROD MODE: Converting {len(non_ocr_files)} PDFs ===")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")
    else:
        print("Mode: Overwrite in place")

    for i, input_path in enumerate(non_ocr_files, 1):
        rel_path = os.path.relpath(input_path, config["directory"])

        if output_dir:
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
        else:
            output_path = input_path

        print(f"[{i}/{len(non_ocr_files)}] {rel_path} ... ", end="", flush=True)
        status, msg = ocr_pdf(input_path, output_path, languages, ocrmypdf_options, overwrite, skip_existing)

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
