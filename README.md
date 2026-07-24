# OCRDAC — OCR Detection And Conversion

Scans directories for PDFs, detects which need OCR, and converts them using a
**dual-image pipeline** that preserves the original visual layer pixel-for-pixel
while applying preprocessing only to the OCR input image.

## Features

- **Dual-image pipeline** — OCR runs on a preprocessed copy; the output PDF
  retains the original crisp visuals.
- **Auto-preprocessing detection** — analyzes each page for low contrast,
  horizontal stripe artifacts, and uneven background; applies median filtering
  + threshold only where needed.
- **Already-OCR'd file handling** — copies already-OCR'd PDFs to the output
  directory without re-processing.
- **Preprocessing statistics** — per-run summary of clean vs. preprocessed
  pages and the reasons preprocessing was triggered.
- **Progress reporting** — dot-based progress during scanning and per-file
  status during conversion.
- **Metadata injection** — writes version, preprocessing decisions, and
  timestamps into each output PDF.

## Requirements

### Python packages

```
pip install Pillow pikepdf
```

### System tools (must be on `PATH`)

| Tool | Purpose |
|---|---|
| [ocrmypdf](https://ocrmypdf.readthedocs.io/) | OCR engine wrapper |
| [Ghostscript](https://ghostscript.com/) (`gs`) | PDF rendering & assembly |
| [Tesseract](https://github.com/tesseract-ocr/tesseract) | OCR engine (used by ocrmypdf) |

Install on Debian/Ubuntu:

```bash
sudo apt install ocrmypdf ghostscript tesseract-ocr
```

## Usage

```bash
python3 OCRDAC.py [directory] [mode]
```

### Arguments

| Position | Argument | Description |
|---|---|---|
| 1 | `directory` | Path to scan for PDFs (overrides config) |
| 2 | `mode` | `dry-run` or `prod` (overrides config) |

### Modes

- **dry-run** — Scan only: list PDFs that need OCR vs. already have text.
  Writes `ocr_files.txt` and `non_ocr_files.txt`.
- **prod** — Scan + convert: copies already-OCR'd PDFs to the output
  directory and runs the dual-image OCR pipeline on PDFs that need it.

### Examples

```bash
# Scan default directory in dry-run mode
python3 OCRDAC.py

# Scan a specific directory
python3 OCRDAC.py ./my_pdfs dry-run

# Full production run
python3 OCRDAC.py ./my_pdfs prod
```

## Configuration

Copy or edit `ocrdac.config` in the project directory:

```ini
[OCRDAC]
mode = prod
directory = ./original_pdf_files
output_dir = ./ocred_output_files
ocr_languages = eng
auto_preprocessing = true
preprocessing = none
median_filter_size = 3
threshold = 130
```

Key settings:

| Setting | Default | Description |
|---|---|---|
| `mode` | `dry-run` | `dry-run` or `prod` |
| `directory` | `./pdfs` | Directory to scan |
| `output_dir` | *(empty)* | Output directory (empty = overwrite in place) |
| `ocr_languages` | `eng` | Tesseract language(s) |
| `auto_preprocessing` | `true` | Enable per-page preprocessing detection |
| `preprocessing` | `none` | Manual override: `none` or `median` |
| `median_filter_size` | `3` | Median filter kernel size (odd) |
| `threshold` | `130` | Binarization threshold (0–255) |

## Pipeline

For each page of a PDF that needs OCR:

1. **Extract** the original page image — never modified.
2. **Create** an OCR-only copy.
3. **If auto-preprocessing** detects low contrast, stripes, or uneven
   background, apply median filter + threshold to the OCR copy.
4. **Feed only** the OCR copy to OCRmyPDF (`--force-ocr --image-dpi 300`).
5. **Replace** the image layer in the OCRmyPDF output with the original
   image, preserving crisp visuals.
6. **Write** provenance metadata (version, preprocessing decisions, date).

## Output

```
==================================================
OCRDAC SUMMARY
==================================================
Time elapsed: 00:15:50
Total processing time: 950s
Files scanned: 58
  Already OCR'd: 10
  Need OCR:      48

Conversion results:
  Successful: 48
  Skipped:    0
  Errors:     0

Auto-Preprocessing Summary:
  Clean pages (no preprocessing needed): 59
  Preprocessed pages: 28
  Reasons:
    - stripe_artifacts: 28

Results saved to:
  OCR'd files:     ocr_files.txt
  Non-OCR files:   non_ocr_files.txt
```

## Project Structure

```
OCRDAC.py                  — Main entry point and pipeline
preprocessing_detector.py  — Per-page preprocessing detection
ocrdac.config              — User configuration (INI)
tests/                     — pytest test suite
  test_console_output.py
  test_dual_image_pipeline.py
  test_end_to_end.py
  test_metadata.py
  test_new_behaviors.py
  test_preprocessing_detector.py
original_pdf_files/        — Source PDFs
ocred_output_files/        — Converted PDFs
ocr_files.txt              — Files already having text (generated)
non_ocr_files.txt          — Files needing OCR (generated)
```

## Running Tests

```bash
python3 -m pytest tests/
```

Some tests require ocrmypdf, Ghostscript, and Tesseract and will be skipped
if those tools are not available.

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
