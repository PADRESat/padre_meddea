"""Test for the fits_tools module"""

import json
import pytest
from pathlib import Path

import astropy.io.fits as fits
from astropy.table import Table
from solarnet_metadata.schema import SOLARNETSchema

import padre_meddea
from padre_meddea.io.fits_tools import *


def tests_get_padre_schema():
    """Test getting SOLARNET Schema with PADRE Overrides"""
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])
    assert isinstance(schema.default_attributes, dict)


def test_comment_lookup_hdr0():
    """Test that all keywords in default HDU Keywords can have comments retrieved"""
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])
    for attr in schema.default_attributes:
        comment = get_comment(attr)
        assert comment is not None
        assert isinstance(comment, str)


def test_get_primary_header():
    test_file_path = Path("test.fits")
    test_data_level = "L0"
    test_data_type = "photon"
    primary_hdr = get_primary_header(test_file_path, test_data_level, test_data_type)
    assert isinstance(primary_hdr, fits.Header)
    assert len(primary_hdr) > 0
    assert "DATE" in primary_hdr
    assert "LEVEL" in primary_hdr
    assert primary_hdr["LEVEL"] == test_data_level
    assert "BTYPE" in primary_hdr
    assert primary_hdr["BTYPE"] == test_data_type
    assert "ORIGAPID" in primary_hdr
    assert primary_hdr["ORIGAPID"] == padre_meddea.APID[test_data_type]
    assert "ORIGFILE" in primary_hdr
    assert primary_hdr["ORIGFILE"] == test_file_path.name

    # Test Process Info
    processing_keywords = ["PRSTEP", "PRPROC", "PRPVER", "PRLIB", "PRVER", "PRHSH"]
    assert all(
        any(this_keyword in this_card.keyword for this_card in primary_hdr.cards)
        for this_keyword in processing_keywords
    )


@pytest.mark.parametrize(
    "test_input,expected",
    [
        ("PRSTEP2", "Processing step type"),
        ("PRSTEP1", "Processing step type"),
        ("PRPROC3", "Name of procedure performing PRSTEP3"),
        ("PRHSH5A", "GIT commit hash for PRLIB5A"),
    ],
)
def test_get_std_comment(test_input, expected):
    assert get_comment(test_input) == expected


def verify_fits_file(test_file, reference_files=None):
    actual_counts = {}
    expected_counts = {}
    test_file = Path(test_file)

    if reference_files:
        if isinstance(reference_files, (str, Path)):
            reference_files = [reference_files]

        for ref_file in reference_files:
            ref_file = Path(ref_file)
            with fits.open(ref_file) as hdul:
                for hdu in hdul:
                    if isinstance(hdu, fits.BinTableHDU):
                        table = Table.read(hdu)
                        expected_counts[hdu.name] = expected_counts.get(
                            hdu.name, 0
                        ) + len(table)

    with fits.open(test_file) as hdul:
        total_actual = 0
        for hdu in hdul:
            if isinstance(hdu, fits.BinTableHDU):
                table = Table.read(hdu)
                actual_len = len(table)
                actual_counts[hdu.name] = actual_len
                total_actual += actual_len

        if expected_counts:
            total_expected = sum(expected_counts.values())
            assert (
                total_actual == total_expected
            ), f"Total rows mismatch: {total_actual} != {total_expected}"
            for name, count in actual_counts.items():
                if name in expected_counts:
                    assert count == expected_counts[name], f"HDU {name} row mismatch"
                else:
                    raise AssertionError(f"Unexpected HDU {name} in output")
    return True


@pytest.fixture
def fits_input_files():
    fits_file1 = (
        Path(__file__).parent.parent
        / "data"
        / "test"
        / "padre_meddea_l1test_eventlist_20250504T055311_v0.1.0.fits"
    )
    fits_file2 = (
        Path(__file__).parent.parent
        / "data"
        / "test"
        / "padre_meddea_l1test_eventlist_20250504T093905_v0.1.0.fits"
    )

    return [fits_file1, fits_file2]


@pytest.fixture
def extra_fits_file():
    return (
        Path(__file__).parent.parent
        / "data"
        / "test"
        / "padre_meddea_l1test_eventlist_20250504T105954_v0.1.0.fits"
    )


def test_concatenate_fits_basic(fits_input_files):
    output_file = concatenate_daily_fits(fits_input_files)
    assert output_file.exists()
    verify_fits_file(output_file, reference_files=fits_input_files)

    # Verify PARENTXT keyword
    with fits.open(output_file) as hdul:
        primary_header = hdul[0].header
        assert "PARENTXT" in primary_header
        assert (
            primary_header["PARENTXT"]
            == "padre_meddea_l1test_eventlist_20250504T055311_v0.1.0.fits, padre_meddea_l1test_eventlist_20250504T093905_v0.1.0.fits"
        )

    # Verify COMMENT in primary header and json load
    with fits.open(output_file) as hdul:
        primary_header = hdul[0].header
        assert "COMMENT" in primary_header
        comment_raw = primary_header.get("COMMENT", "")
        if isinstance(comment_raw, list):
            comment_str = "".join(comment_raw)  # Avoid actual newlines
        else:
            comment_str = str(comment_raw).replace("\n", "")

        file_time_list = json.loads(comment_str)

        assert isinstance(file_time_list, list)
        assert len(file_time_list) == 2
        assert file_time_list[0] == {
            "date-beg": "2025-05-04T05:53:11.353",
            "date-end": "2025-05-04T07:37:49.299",
            "filename": "padre_meddea_l1test_eventlist_20250504T055311_v0.1.0.fits",
        }


def test_concatenate_fits_with_existing(fits_input_files, extra_fits_file):
    # Test appending to an existing file
    existing_file = concatenate_daily_fits([fits_input_files[0]])

    output_file = concatenate_daily_fits(
        [fits_input_files[1], extra_fits_file], existing_file=existing_file
    )
    assert output_file.exists()
    verify_fits_file(
        output_file,
        reference_files=[fits_input_files[0], fits_input_files[1], extra_fits_file],
    )

    # Verify PARENTXT keyword
    with fits.open(output_file) as hdul:
        primary_header = hdul[0].header
        assert "PARENTXT" in primary_header
        assert (
            primary_header["PARENTXT"]
            == "padre_meddea_l1test_eventlist_20250504T055311_v0.1.0.fits, padre_meddea_l1test_eventlist_20250504T093905_v0.1.0.fits, padre_meddea_l1test_eventlist_20250504T105954_v0.1.0.fits"
        )

    # Verify COMMENT in primary header and json load
    with fits.open(output_file) as hdul:
        primary_header = hdul[0].header
        assert "COMMENT" in primary_header
        comment_raw = primary_header.get("COMMENT", "")

        if isinstance(comment_raw, list):
            comment_str = "".join(comment_raw)  # Avoid actual newlines
        else:
            comment_str = str(comment_raw).replace("\n", "")
        log.info(f"COMMENT: {comment_str}")
        file_time_list = json.loads(comment_str)

        assert isinstance(file_time_list, list)
        assert len(file_time_list) == 3
