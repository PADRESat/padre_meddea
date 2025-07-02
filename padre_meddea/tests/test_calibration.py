from pathlib import Path

import pytest

from astropy.io import fits

import padre_meddea
import padre_meddea.calibration.calibration as calib


@pytest.mark.parametrize(
    "bin_file,expected_data_type",
    [
        ("apid160_4packets.bin", "photon"),
        ("apid162_4packets.bin", "spectrum"),
        ("apid163_4packets.bin", "housekeeping"),
    ],
)
def test_process_file_test_files(bin_file, expected_data_type):
    files = calib.process_file(
        padre_meddea._test_files_directory / bin_file, overwrite=True
    )
    assert Path(files[0]).exists()
    with fits.open(files[0]) as f:
        assert f[0].header["INSTRUME"] == "MeDDEA"

    # Check that the filename includes the correct data type
    assert f"padre_meddea_l0test_{expected_data_type}_" in files[0]
    assert files[0].endswith(".fits")

    # Clean up
    Path(files[0]).unlink()
