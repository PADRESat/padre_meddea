import json
import os
import platform
import pytest
from pathlib import Path
import tempfile
import time

import astropy.io.fits as fits
import pytest

from padre_meddea.io.fits_tools import concatenate_files


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
    "input_files, expected_outputs, expected_parentxt, additional_file, expected_additional_outputs, additional_parentext, expected_first_comment",
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
            {
                "date-beg": "2025-05-04T05:53:11.353",
                "date-end": "2025-05-04T05:53:11.830",
                "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
            },
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
            {
                "date-beg": "2025-05-04T05:53:11.353",
                "date-end": "2025-05-04T05:53:11.830",
                "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
            },
        ),
        # Corrected HK case
        (
            [
                data_dir / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
                data_dir / "hk/padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
                data_dir / "hk/padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            [
                data_dir / "hk/padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v0.1.0.fits",
                "padre_meddea_l1_housekeeping_20250505T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            {
                "date-beg": "2025-05-0405:51:38.000",
                "date-end": "2025-05-04T05:53:08.000",
                "filename": "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
            },
        ),
        # Corrected spec case
        (
            [
                data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits",
                data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits",
                data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits",
            [
                data_dir / "spec/padre_meddea_l0test_spectrum_20250504T153709_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v0.1.0.fits",
                "padre_meddea_l1_spectrum_20250505T000000_v0.1.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153309_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153509_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T153709_v0.1.0.fits",
            {
                "date-beg": "2025-05-04T15:31:11.449",
                "date-end": "2025-05-04T15:33:09.809",
                "filename": "padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits",
            },
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
    expected_first_comment,
):
    os.chdir(tmpdir)

    output_files = concatenate_files(input_files)
    assert all([output_file.exists() for output_file in output_files])
    assert len(output_files) == len(expected_outputs)
    assert all(
        [
            str(output_files[i]) == expected_outputs[i]
            for i in range(len(expected_outputs))
        ]
    )
    print(f"Finished First Concat with Outputs: {output_files}")

    output_file = output_files[0]

    # Check the primary header contents
    with fits.open(output_file, memmap=False) as hdul:
        header = hdul[0].header

        # assert "PARENTXT" in header
        # assert header["PARENTXT"] == expected_parentxt
        # expected_parentxt_list = expected_parentxt.split(", ")

        # comment_raw = header.get("COMMENT", "")
        # if isinstance(comment_raw, list):
        #     comment_str = "".join(comment_raw)
        # else:
        #     comment_str = str(comment_raw).replace("\n", "")

        # file_time_list = json.loads(comment_str)
        # assert isinstance(file_time_list, list)
        # assert len(file_time_list) == len(expected_parentxt_list)
        # assert file_time_list[0] == expected_first_comment

    # Add additional file checks
    output_files = concatenate_files(additional_file, existing_file=output_file)
    assert all([output_file.exists() for output_file in output_files])
    assert len(output_files) == len(expected_additional_outputs)
    assert all(
        [
            str(output_files[i]) == expected_additional_outputs[i]
            for i in range(len(expected_additional_outputs))
        ]
    )

    output_file = output_files[0]

    # Check the primary header contents again
    with fits.open(output_file, memmap=False) as hdul:
        header = hdul[0].header

        # assert "PARENTXT" in header
        # assert header["PARENTXT"] == additional_parentext
        # expected_parentxt_list = additional_parentext.split(", ")

        # comment_raw = header.get("COMMENT", "")
        # if isinstance(comment_raw, list):
        #     comment_str = "".join(comment_raw)
        # else:
        #     comment_str = str(comment_raw).replace("\n", "")

        # file_time_list = json.loads(comment_str)
        # assert isinstance(file_time_list, list)
        # assert len(file_time_list) == len(expected_parentxt_list)