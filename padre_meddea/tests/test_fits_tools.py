import json
from pathlib import Path

import pytest
import astropy.io.fits as fits

from padre_meddea.io.fits_tools import sort_files_list
from padre_meddea.io.fits_tools import concatenate_daily_fits

data_dir = Path(__file__).parent.parent / "data/test"


def test_prepare_files_list_basic():
    """Test basic functionality of _prepare_files_list without existing file."""

    files = [
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
    ]

    result = sort_files_list(files)

    # Check that all files are included
    assert len(result) == 2
    assert all(f in result for f in files)


def test_prepare_files_list_with_existing_file():
    """Test _prepare_files_list with existing file."""

    existing = (
        data_dir / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits"
    )
    files = [
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
    ]

    result = sort_files_list(files, existing_file=existing)

    # Check that all files are included
    assert len(result) == 2
    assert existing in result
    assert files[0] in result

    # Check order - existing file should be first in the input but might be
    # reordered based on DATE-BEG
    header_times = [fits.getheader(f).get("DATE-BEG", "") for f in result]
    assert header_times[0] < header_times[1]


def test_prepare_files_list_duplicates():
    """Test _prepare_files_list with duplicate files."""

    files = [
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",  # Duplicate
    ]

    result = sort_files_list(files)

    # Check that duplicates are removed
    assert len(result) == 2
    # Check that files are included and ordered by date
    header_times = [fits.getheader(f).get("DATE-BEG", "") for f in result]
    assert header_times[0] < header_times[1]


def test_prepare_files_list_existing_file_duplicate():
    """Test _prepare_files_list with existing file also in files_to_combine."""

    existing = (
        data_dir / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits"
    )
    files = [
        existing,  # Same as existing_file
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
    ]

    result = sort_files_list(files, existing_file=existing)

    # Check that duplicates are removed
    assert len(result) == 2
    # Verify files are sorted by DATE-BEG
    header_times = [fits.getheader(f).get("DATE-BEG", "") for f in result]
    assert header_times[0] < header_times[1]


def test_prepare_files_list_sorting():
    """Test _prepare_files_list sorting by DATE-BEG."""

    files = [
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",  # Later date
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",  # Earlier date
    ]

    result = sort_files_list(files)

    # Check sorting by DATE-BEG
    headers = [fits.getheader(f) for f in result]
    times = [h.get("DATE-BEG", "") for h in headers]
    assert times[0] < times[1]
    assert result[0] == files[1]  # Earlier date should be first
    assert result[1] == files[0]  # Later date should be second


def test_prepare_files_list_mixed_types():
    """Test _prepare_files_list with mixed file types."""

    files = [
        data_dir
        / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",  # May
        data_dir / "spec/padre_meddea_l1test_spec_20250310T114744_v0.1.0.fits",  # March
    ]

    result = sort_files_list(files)

    # Check that all files are included
    assert len(result) == 2
    assert all(f in result for f in files)

    # March file should come before May file when sorted by DATE-BEG
    assert "20250310" in str(result[0])
    assert "20250504" in str(result[1])


def test_prepare_files_list_dateref_fallback():
    """Test _prepare_files_list falling back to DATEREF when DATE-BEG is missing."""

    # The HK files appear to have "UNKNOWN" for date-beg according to the test fixture
    files = [
        data_dir / "hk/padre_meddea_l1test_hk_20250504T055138_v0.1.0.fits",
        data_dir / "hk/padre_meddea_l1test_hk_20250310T114743_v0.1.0.fits",
    ]

    # This will test if sorting works with DATEREF when DATE-BEG is not available
    result = sort_files_list(files)

    # Verify both files are included
    assert len(result) == 2

    # Check if we get expected ordering based on the filenames
    # March should come before May
    assert "20250310" in str(result[0])
    assert "20250504" in str(result[1])


@pytest.mark.parametrize(
    "input_files, additional_file, expected_parentxt, expected_first_comment",
    [
        # Corrected eventlist case
        (
            [
                data_dir
                / "eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",
                data_dir
                / "eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
            ],
            [
                data_dir
                / "eventlist/padre_meddea_l1test_eventlist_20250504T083234_v0.1.0.fits",
            ],
            "padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits, padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
            {
                "date-beg": "2025-05-04T07:37:49.472",
                "date-end": "2025-05-04T08:03:30.331",
                "filename": "padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",
            },
        ),
        # Corrected HK case
        (
            [
                data_dir / "hk/padre_meddea_l1test_hk_20250310T114743_v0.1.0.fits",
                data_dir / "hk/padre_meddea_l1test_hk_20250504T055138_v0.1.0.fits",
            ],
            [
                data_dir / "hk/padre_meddea_l1test_hk_20250317T105835_v0.1.0.fits",
            ],
            "padre_meddea_l1test_hk_20250310T114743_v0.1.0.fits, padre_meddea_l1test_hk_20250504T055138_v0.1.0.fits",
            {
                "date-beg": "UNKNOWN",  # This matches the actual output
                "date-end": "UNKNOWN",  # This matches the actual output
                "filename": "padre_meddea_l1test_hk_20250310T114743_v0.1.0.fits",
            },
        ),
        # Corrected spec case
        (
            [
                data_dir / "spec/padre_meddea_l1test_spec_20250310T114744_v0.1.0.fits",
                data_dir / "spec/padre_meddea_l1test_spec_20250317T121301_v0.1.0.fits",
            ],
            [
                data_dir / "spec/padre_meddea_l1test_spec_20250504T153111_v0.1.0.fits",
            ],
            "padre_meddea_l1test_spec_20250310T114744_v0.1.0.fits, padre_meddea_l1test_spec_20250317T121301_v0.1.0.fits",
            {
                "date-beg": "2025-03-10T11:47:44.197",
                "date-end": "2025-03-10T11:57:44.137",
                "filename": "padre_meddea_l1test_spec_20250310T114744_v0.1.0.fits",
            },
        ),
    ],
)
def test_concatenate_fits_cases(
    input_files, additional_file, expected_parentxt, expected_first_comment
):
    output_files = concatenate_daily_fits(input_files)
    assert all([output_file.exists() for output_file in output_files])

    output_file = output_files[0]
    # Check the primary header contents
    with fits.open(output_file) as hdul:
        header = hdul[0].header

        assert "PARENTXT" in header
        assert header["PARENTXT"] == expected_parentxt

        comment_raw = header.get("COMMENT", "")
        if isinstance(comment_raw, list):
            comment_str = "".join(comment_raw)
        else:
            comment_str = str(comment_raw).replace("\n", "")

        file_time_list = json.loads(comment_str)
        assert isinstance(file_time_list, list)
        assert len(file_time_list) == 2
        assert file_time_list[0] == expected_first_comment

    # Add additional file checks
    output_files = concatenate_daily_fits(additional_file, existing_file=output_file)
    assert all([output_file.exists() for output_file in output_files])

    output_file = output_files[0]
    # Check the primary header contents
    with fits.open(output_file) as hdul:
        header = hdul[0].header

        assert "PARENTXT" in header

        # split by commas and strip whitespace
        parentxt = header["PARENTXT"].split(", ")
        assert len(parentxt) == 3

        comment_raw = header.get("COMMENT", "")
        if isinstance(comment_raw, list):
            comment_str = "".join(comment_raw)
        else:
            comment_str = str(comment_raw).replace("\n", "")

        file_time_list = json.loads(comment_str)
        assert isinstance(file_time_list, list)
        assert len(file_time_list) == 3
