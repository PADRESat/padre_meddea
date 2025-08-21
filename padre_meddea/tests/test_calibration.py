from pathlib import Path

import pytest
from astropy.io import fits
from astropy.time import Time

import padre_meddea
import padre_meddea.calibration.calibration as calib
import padre_meddea.io.file_tools as file_tools


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
        padre_meddea._test_files_directory / bin_file, overwrite=False
    )
    assert Path(files[0]).exists()
    with fits.open(files[0]) as f:
        assert f[0].header["INSTRUME"] == "MeDDEA"

    # Check that the filename includes the correct data type
    assert f"padre_meddea_l0_{expected_data_type}_" in files[0].name
    assert files[0].name.endswith(".fits")

    match expected_data_type:
        case "photon":
            photon_list = file_tools.read_fits_l0l1_photon(files[0])
            pkt_list, event_list = photon_list.pkt_list, photon_list.event_list
            assert all(pkt_list.time > Time("2024-01-01T00:00"))
            assert all(event_list.time > Time("2024-01-01T00:00"))
        case "spectrum":
            spectrum_list = file_tools.read_fits_l0l1_spectrum(files[0])
            assert all(spectrum_list.time > Time("2024-01-01T00:00"))
        case "housekeeping":
            hk_ts, cmd_ts = file_tools.read_fits_l0l1_housekeeping(files[0])
            assert all(hk_ts.time > Time("2024-01-01T00:00"))
            if "time" in cmd_ts.colnames:  # Check if command response times are present
                # If command response times are present, check them
                assert all(cmd_ts.time > Time("2024-01-01T00:00"))

    # Clean up
    Path(files[0]).unlink()
