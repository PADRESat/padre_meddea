import json
import os
import platform
import pytest
from pathlib import Path
import tempfile
import time
import numpy as np

from astropy.table import Table
import astropy.io.fits as fits
import pytest

from padre_meddea.io.fits_tools import (
    concatenate_files,
    hdu_to_dict,
    get_hdu_data_times,
)


# Fix for memmap issue and files not closing on Windows
@pytest.fixture(autouse=True)
def patch_fits_open_for_windows(monkeypatch):
    """Ensure fits.open is called with memmap=False on Windows."""
    if platform.system().lower() == "windows":
        original_open = fits.open

        def patched_open(*args, **kwargs):
            kwargs.setdefault("memmap", False)
            return original_open(*args, **kwargs)

        monkeypatch.setattr(fits, "open", patched_open)


data_dir = Path(__file__).parent.parent / "data/test"


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.mark.parametrize(
    "input_files, expected_outputs, expected_parentxt, additional_file, expected_additional_outputs, additional_parentext, expected_provenance_rows",
    [
        # Corrected eventlist (single-day) case
        (
            [
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v0.1.0.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
            [
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v0.1.0.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:53:11.353",
                    "date-end": "2025-05-04T05:53:11.830",
                    "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T07:37:49.472",
                    "date-end": "2025-05-04T07:37:49.476",
                    "filename": "padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:03:30.385",
                    "date-end": "2025-05-04T08:03:30.390",
                    "filename": "padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
                },
            ],
        ),
        # Corrected eventlist (Multi-day) case
        (
            [
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v0.1.0.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            [
                data_dir
                / "eventlist/padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_photon_20250504T000000_v0.1.0.fits",
                "padre_meddea_l1_photon_20250505T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits, padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:53:11.353",
                    "date-end": "2025-05-04T05:53:11.830",
                    "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T07:37:49.472",
                    "date-end": "2025-05-04T07:37:49.476",
                    "filename": "padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:03:30.385",
                    "date-end": "2025-05-04T08:03:30.390",
                    "filename": "padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:32:34.299",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
                },
            ],
        ),
        # Corrected HK case
        (
            [
                data_dir
                / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
                data_dir
                / "hk/padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
                data_dir
                / "hk/padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            [
                data_dir
                / "hk/padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v0.1.0.fits",
                "padre_meddea_l1_housekeeping_20250505T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:51:38.000",
                    "date-end": "2025-05-04T05:53:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:53:38.000",
                    "date-end": "2025-05-04T05:55:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:55:38.000",
                    "date-end": "2025-05-04T05:57:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:57:38.000",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
                },
            ],
        ),
        # Corrected spec case
        (
            [
                data_dir
                / "spec/padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits",
                data_dir
                / "spec/padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits",
                data_dir
                / "spec/padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits",
            [
                data_dir
                / "spec/padre_meddea_l0test_spectrum_20250504T153709_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v0.1.0.fits",
                "padre_meddea_l1_spectrum_20250505T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153709_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T15:31:11.449",
                    "date-end": "2025-05-04T15:33:09.809",
                    "filename": "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T15:33:11.449",
                    "date-end": "2025-05-04T15:35:09.809",
                    "filename": "padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T15:35:11.449",
                    "date-end": "2025-05-04T15:37:09.809",
                    "filename": "padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T15:37:11.449",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_spectrum_20250504T153709_v0.1.0.fits",
                },
            ],
        ),
    ],
)
def test_concatenate_fits_cases(
    tmpdir,
    input_files,
    expected_outputs,
    expected_parentxt,
    additional_file,
    expected_additional_outputs,
    additional_parentext,
    expected_provenance_rows,
):
    os.chdir(tmpdir)

    # Initial concatenation
    output_files = concatenate_files(input_files)
    assert all(output_file.exists() for output_file in output_files)
    assert len(output_files) == len(expected_outputs)
    for output_file, expected_name in zip(output_files, expected_outputs):
        assert output_file.name == expected_name

    output_file = output_files[0]
    expected_parentxt_list = expected_parentxt.split(", ")

    # Check primary header and provenance
    with fits.open(output_file, memmap=False) as hdul:
        header = hdul[0].header
        assert "PARENTXT" in header
        assert header["PARENTXT"] == expected_parentxt

        provenance_hdu = next((hdu for hdu in hdul if hdu.name == "PROVENANCE"), None)
        assert provenance_hdu is not None, "PROVENANCE HDU not found"

        provenance_table = Table(provenance_hdu.data)
        assert len(provenance_table) == len(expected_parentxt_list)
        assert {"FILENAME", "DATE_BEG", "DATE_END"} <= set(provenance_table.colnames)

        # Handle both dict or list of dicts
        assert (
            provenance_table[0]["FILENAME"] == expected_provenance_rows[0]["filename"]
        )
        assert (
            str(provenance_table[0]["DATE_BEG"])
            == expected_provenance_rows[0]["date-beg"]
        )
        assert (
            str(provenance_table[0]["DATE_END"])
            == expected_provenance_rows[0]["date-end"]
        )

    # Add additional file and re-validate
    output_files = concatenate_files(additional_file, existing_file=output_file)
    assert all(output_file.exists() for output_file in output_files)
    assert len(output_files) == len(expected_additional_outputs)
    for output_file, expected_name in zip(output_files, expected_additional_outputs):
        assert output_file.name == expected_name

    output_file = output_files[0]

    with fits.open(output_file, memmap=False) as hdul:
        header = hdul[0].header
        assert "PARENTXT" in header
        assert header["PARENTXT"] == additional_parentext

        provenance_hdu = next((hdu for hdu in hdul if hdu.name == "PROVENANCE"), None)
        assert provenance_hdu is not None, "PROVENANCE HDU not found after append"

        provenance_table = Table(provenance_hdu.data)
        additional_parentxt_list = additional_parentext.split(", ")
        assert len(provenance_table) == len(additional_parentxt_list)

        expected_rows = []

        for i, filename in enumerate(additional_parentxt_list):
            assert provenance_table["FILENAME"][i] == filename
            assert provenance_table["DATE_BEG"][i] is not None
            assert provenance_table["DATE_END"][i] is not None

            expected_row = {
                "filename": provenance_table["FILENAME"][i],
                "date-beg": provenance_table["DATE_BEG"][i],
                "date-end": provenance_table["DATE_END"][i],
            }

            expected_rows.append(expected_row)

        assert (
            expected_rows == expected_provenance_rows
        ), "Expected rows do not match actual rows in provenance table"

    if len(output_files) > 1:
        secondary_output_file = output_files[1]

        with fits.open(secondary_output_file, memmap=False) as hdul:
            header = hdul[0].header
            assert "PARENTXT" in header

            # Get last item in PARENTXT since that is the multi-date test case
            additional_parentext = additional_parentext.split(", ")[-1]

            assert header["PARENTXT"] == additional_parentext

            provenance_hdu = next(
                (hdu for hdu in hdul if hdu.name == "PROVENANCE"), None
            )
            assert provenance_hdu is not None, "PROVENANCE HDU not found after append"

            provenance_table = Table(provenance_hdu.data)
            additional_parentxt_list = additional_parentext.split(", ")
            assert len(provenance_table) == 1
            assert {"FILENAME", "DATE_BEG", "DATE_END"} <= set(
                provenance_table.colnames
            )
            assert provenance_table["FILENAME"][0] == additional_parentext
            assert provenance_table["DATE_BEG"][0] == "2025-05-05T00:00:00.000"


@pytest.mark.parametrize(
    "input_files",
    [
        [
            data_dir
            / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
            data_dir
            / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
        ],
        [
            data_dir
            / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
            data_dir
            / "hk/padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
        ],
        [
            data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits",
            data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits",
        ],
    ],
)
def test_eventlist_concatenate(input_files):
    """Test that no data is lost in every hdu and that the data is in the right order."""

    output_files = concatenate_files(files_to_combine=input_files)

    h1 = fits.open(input_files[0])
    h2 = fits.open(input_files[1])
    out = fits.open(output_files[0])

    # check all hdu but ignore the first hdu which does not contain data
    for hdu1, hdu2, outhdu in zip(h1[1:], h2[1:], out[1:]):
        assert len(hdu1.data) + len(hdu2.data) == len(outhdu.data)

        times = get_hdu_data_times(hdu_to_dict(outhdu))
        assert np.all(sorted(times) == times)

    h1.close()
    h2.close()
    out.close()


# add test to check that no data was lost for each file
# add test to check that data is in the right order
