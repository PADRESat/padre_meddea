from pathlib import Path

import pytest

from astropy.io import fits

import padre_meddea
import padre_meddea.calibration.calibration as calib


def test_process_file_test_file():
    files = calib.process_file(
        padre_meddea._test_files_directory / "apid160_4packets.bin", overwrite=True
    )
    assert Path(files[0]).exists
    with fits.open(files[0]) as f:
        assert f[0].header["INSTRUME"] == "MeDDEA"
        # assert f[1].data["atod"][0] == 1336
        # assert len(f[1].data["atod"]) == 760
    Path(files[0]).unlink()

    # Assert filename is correct
    assert files[0] == "padre_meddea_l0test_eventlist_20240916T122901_v0.1.0.fits"
