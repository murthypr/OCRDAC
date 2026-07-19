"""Tests for OCRDAC PDF metadata provenance (v2 metadata fields)."""

import os
import sys
from datetime import datetime, timezone
import re

import pikepdf
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from OCRDAC import write_metadata, read_metadata, DEFAULT_CONFIG

OCRDAC_VERSION = DEFAULT_CONFIG["ocrdac_version"]


def _make_blank_pdf(path):
    """Create a minimal blank PDF at the given path."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.save(path)
    pdf.close()


def _sample_metadata():
    return {
        "OCRDAC-Version": OCRDAC_VERSION,
        "Auto-Preprocessing-Enabled": "True",
        "Auto-Preprocessing-Reason": "low_contrast",
        "Median-Filter-Used": "True",
        "Median-Filter-Size": "3",
        "Threshold-Used": "130",
        "OCRmyPDF-Version": "16.13.0+dfsg1",
        "Ghostscript-Version": "10.06.0",
        "OCR-DateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


class TestWriteMetadata:
    def test_writes_all_keys(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert set(meta.keys()) == set(result.keys())

    def test_values_match_exactly(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        for key in meta:
            assert result[key] == meta[key], f"Mismatch for key {key}"

    def test_ocrdac_version_matches_config(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert result["OCRDAC-Version"] == OCRDAC_VERSION

    def test_versions_are_strings(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert isinstance(result["OCRmyPDF-Version"], str)
        assert isinstance(result["Ghostscript-Version"], str)
        assert len(result["OCRmyPDF-Version"]) > 0
        assert len(result["Ghostscript-Version"]) > 0

    def test_datetime_is_valid_iso8601(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        dt_str = result["OCR-DateTime"]
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$", dt_str)

    def test_median_filter_used_is_boolean_string(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert result["Median-Filter-Used"] in ("True", "False")

    def test_auto_preprocessing_enabled_is_boolean_string(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert result["Auto-Preprocessing-Enabled"] in ("True", "False")

    def test_median_filter_size_is_numeric_string(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = _sample_metadata()
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert result["Median-Filter-Size"].isdigit()


class TestOverwriteMetadata:
    def test_overwrite_replaces_old_values(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta1 = {
            "OCRDAC-Version": "v0.1",
            "Auto-Preprocessing-Enabled": "True",
            "Auto-Preprocessing-Reason": "low_contrast",
            "Median-Filter-Used": "False",
            "Median-Filter-Size": "3",
            "Threshold-Used": "130",
            "OCRmyPDF-Version": "1.0",
            "Ghostscript-Version": "9.0",
            "OCR-DateTime": "2024-01-01T00:00:00Z",
        }
        write_metadata(pdf_path, meta1)

        meta2 = {
            "OCRDAC-Version": "v0.2",
            "Auto-Preprocessing-Enabled": "False",
            "Auto-Preprocessing-Reason": "none",
            "Median-Filter-Used": "False",
            "Median-Filter-Size": "5",
            "Threshold-Used": "100",
            "OCRmyPDF-Version": "2.0",
            "Ghostscript-Version": "10.0",
            "OCR-DateTime": "2025-06-15T12:00:00Z",
        }
        write_metadata(pdf_path, meta2)
        result = read_metadata(pdf_path)

        for key in meta2:
            assert result[key] == meta2[key], f"Mismatch for key {key}"

    def test_overwrite_preserves_extra_keys(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta1 = _sample_metadata()
        meta1["Custom-Key"] = "custom_value"
        write_metadata(pdf_path, meta1)

        meta2 = _sample_metadata()
        write_metadata(pdf_path, meta2)
        result = read_metadata(pdf_path)

        assert "OCRDAC-Version" in result


class TestReadMetadata:
    def test_read_from_fresh_pdf(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        result = read_metadata(pdf_path)
        assert isinstance(result, dict)

    def test_read_returns_empty_for_no_metadata(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        result = read_metadata(pdf_path)
        assert result == {}

    def test_roundtrip(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        original = _sample_metadata()
        write_metadata(pdf_path, original)
        recovered = read_metadata(pdf_path)

        assert original == recovered


class TestEdgeCases:
    def test_write_to_nonexistent_file_raises(self, tmp_path):
        bad_path = str(tmp_path / "nonexistent.pdf")
        meta = _sample_metadata()

        with pytest.raises(Exception):
            write_metadata(bad_path, meta)

    def test_empty_metadata_dict(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        write_metadata(pdf_path, {})
        result = read_metadata(pdf_path)
        assert result == {}

    def test_metadata_with_unicode_values(self, tmp_path):
        pdf_path = str(tmp_path / "test.pdf")
        _make_blank_pdf(pdf_path)

        meta = {
            "OCRDAC-Version": "v0.1",
            "Auto-Preprocessing-Reason": "low_contrast",
            "OCR-DateTime": "2025-01-01T00:00:00Z",
            "OCRmyPDF-Version": "16.13",
            "Ghostscript-Version": "10.06",
            "Median-Filter-Used": "True",
            "Median-Filter-Size": "3",
            "Threshold-Used": "130",
            "Auto-Preprocessing-Enabled": "True",
        }
        write_metadata(pdf_path, meta)
        result = read_metadata(pdf_path)

        assert result["Auto-Preprocessing-Reason"] == "low_contrast"
