import subprocess
import os
import sys
import time

OUTPUT_FILE = "non_ocr_files.txt"

def pdf_needs_ocr(path):
    # Returns True if PDF has NO text layer
    result = subprocess.run(
        ["ocrmypdf", "--is-text-visible", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    return result.returncode == 6  # 6 = no text layer

def scan_directory(root_dir):
    root_dir = os.path.abspath(root_dir)
    results = []

    for dirpath, _, filenames in os.walk(root_dir):
        for name in filenames:
            if name.lower().endswith(".pdf"):
                fullpath = os.path.join(dirpath, name)
                print(f"Checking: {fullpath}")

                if pdf_needs_ocr(fullpath):
                    print(f"  → Needs OCR")
                    results.append(fullpath)
                else:
                    print(f"  → Already OCR'd")

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 find_non_ocr.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    start_time = time.perf_counter()
    non_ocr_files = scan_directory(directory)

    # Write results to file
    with open(OUTPUT_FILE, "w") as f:
        for path in non_ocr_files:
            f.write(path + "\n")

    end_time = time.perf_counter()
    elapsed_ms = int((end_time - start_time) * 1000)

    print("\nDone.")
    print(f"Found {len(non_ocr_files)} non-OCR'd PDFs in {elapsed_ms} ms.")
    print(f"List written to: {OUTPUT_FILE}")
