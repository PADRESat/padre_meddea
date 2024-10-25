"""Test for the fits_tools module"""

import pytest

import astropy.io.fits as fits

from padre_meddea.io.fits_tools import *


def test_comment_lookup_hdr0():
    """Test that all keywords in fits_keyword_primaryhdu are listed in fits_keyword_dict"""
    hdr0_keywords = list(FITS_HDR0["keyword"])
    keyword_to_comment = list(FITS_HDR_KEYTOCOMMENT["keyword"])
    for this_keyword in hdr0_keywords:
        assert this_keyword in keyword_to_comment


def test_get_primary_header():
    assert isinstance(get_primary_header(), fits.Header)


def test_add_process_info_to_header():
    """Test that new header cards are added."""
    header = get_primary_header()
    orig_header = header.copy()
    header = add_process_info_to_header(header)
    # check that keywords were added
    assert len(header) > len(orig_header)
    orig_keywords = [this_keyword for this_keyword in orig_header]
    # check that the cards that were added have content
    for this_card in header.cards:
        if this_card.keyword not in orig_keywords:
            assert len(this_card.value) > 0
            assert len(this_card.comment) > 0


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
    assert get_std_comment(test_input) == expected
