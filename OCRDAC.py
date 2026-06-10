import subprocess
import os
import sys
import time

OUTPUT_FILE_OCR = "ocr_files.txt"
OUTPUT_FILE_NON_OCR = "non_ocr_files.txt"

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
    processed_count = 0
    ocr_count = 0
    non_ocr_count = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    ocr_output_path = os.path.join(script_dir, OUTPUT_FILE_OCR)
    non_ocr_output_path = os.path.join(script_dir, OUTPUT_FILE_NON_OCR)

    with open(ocr_output_path, "a", encoding="utf-8") as ocr_file, open(non_ocr_output_path, "a", encoding="utf-8") as non_ocr_file:
        for dirpath, _, filenames in os.walk(root_dir):
            for name in filenames:
                if name.lower().endswith(".pdf"):
                    fullpath = os.path.join(dirpath, name)
                    print(f"Checking: {fullpath}")

                    if pdf_needs_ocr(fullpath):
                        print(f"  → Needs OCR")
                        results.append(fullpath)
                        non_ocr_file.write(fullpath + "\n")
                        non_ocr_file.flush()
                        non_ocr_count += 1
                    else:
                        print(f"  → Already OCR'd")
                        ocr_file.write(fullpath + "\n")
                        ocr_file.flush()
                        ocr_count += 1

                    processed_count += 1
                    print(".", end="", flush=True)
                    if processed_count % 25 == 0:
                        print(f"Processed {processed_count} files. (ocr:{ocr_count} / non-ocr:{non_ocr_count})")

    return results

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 find_non_ocr.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    start_time = time.perf_counter()
    non_ocr_files = scan_directory(directory)
    end_time = time.perf_counter()
    elapsed_seconds = int(end_time - start_time)
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60

    print("\nDone.")
    print(f"Found {len(non_ocr_files)} non-OCR'd PDFs in {hours:02d}:{minutes:02d}:{seconds:02d}.")

    print(f"Results appended to: {OUTPUT_FILE_OCR} and {OUTPUT_FILE_NON_OCR}")
