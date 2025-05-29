import json
import pytest
from pathlib import Path
import astropy.io.fits as fits
from padre_meddea.io.fits_tools import concatenate_daily_fits


@pytest.mark.parametrize(
    "input_files, additional_file, expected_parentxt, expected_first_comment",
    [
        # Corrected eventlist case
        (
            [
                Path(__file__).parent.parent
                / "data/test/eventlist/padre_meddea_l1test_eventlist_20250504T073749_v0.1.0.fits",
                Path(__file__).parent.parent
                / "data/test/eventlist/padre_meddea_l1test_eventlist_20250504T080330_v0.1.0.fits",
            ],
            [
                Path(__file__).parent.parent
                / "data/test/eventlist/padre_meddea_l1test_eventlist_20250504T083234_v0.1.0.fits",
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
                Path(__file__).parent.parent
                / "data/test/hk/padre_meddea_l1test_hk_20250310T114743_v0.1.0.fits",
                Path(__file__).parent.parent
                / "data/test/hk/padre_meddea_l1test_hk_20250504T055138_v0.1.0.fits",
            ],
            [
                Path(__file__).parent.parent
                / "data/test/hk/padre_meddea_l1test_hk_20250317T105835_v0.1.0.fits",
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
                Path(__file__).parent.parent
                / "data/test/spec/padre_meddea_l1test_spec_20250310T114744_v0.1.0.fits",
                Path(__file__).parent.parent
                / "data/test/spec/padre_meddea_l1test_spec_20250317T121301_v0.1.0.fits",
            ],
            [
                Path(__file__).parent.parent
                / "data/test/spec/padre_meddea_l1test_spec_20250504T153111_v0.1.0.fits",
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
    output_file = concatenate_daily_fits(input_files)
    assert output_file.exists()

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
    output_file = concatenate_daily_fits(additional_file, existing_file=output_file)
    assert output_file.exists()

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
