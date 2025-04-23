"""Test for the fits_tools module"""

import pytest

import astropy.io.fits as fits
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
    assert "DATATYPE" in primary_hdr
    assert primary_hdr["DATATYPE"] == test_data_type
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
