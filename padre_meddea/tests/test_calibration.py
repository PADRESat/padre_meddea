from pathlib import Path

import pytest

from astropy.io import fits

import padre_meddea
import padre_meddea.calibration as calib


def test_process_file_test_file():
    files = calib.process_file(
        padre_meddea._test_files_directory / "apid160_4packets.bin", overwrite=True
    )
    assert Path(files[0]).exists
    f = fits.open(files[0])
    assert f[0].header["INSTRUME"] == "MeDDEA"
    # assert f[1].data["atod"][0] == 1336
    # assert len(f[1].data["atod"]) == 760
    Path(files[0]).unlink
