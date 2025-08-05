import astropy.units as u
import numpy as np
import pytest
from astropy.timeseries import TimeSeries

import padre_meddea.housekeeping.calibration as cal
from padre_meddea import _test_files_directory
from padre_meddea.housekeeping import housekeeping as hk

hk_packet_file = _test_files_directory / "apid163_4packets.bin"

NUM_PACKETS = 4


def test_read_hk_file():
    hk_list = hk.parse_housekeeping_packets(hk_packet_file)
    assert len(hk_list) == NUM_PACKETS


def test_hk_packet_definition():
    packet_definition = hk.packet_definition_hk()
    assert (
        len(packet_definition) == len(hk.hk_definitions) + 2
    )  # add one for checksum and one for timestamp


def test_parse_error_summary():
    """Test that this parses things correctly."""
    ts1 = TimeSeries(time_start="2016-03-22T12:30:31", time_delta=3 * u.s, n_samples=16)
    # set each bit in sequence
    ts1["error_summary"] = 2 ** np.arange(16)
    error_ts = cal.parse_error_summary(ts1)
    for i, this_col in enumerate(error_ts.colnames[1:]):
        error_ts[this_col][i] = True


def test_parse_error_summary_raise():
    ts1 = TimeSeries(time_start="2016-03-22T12:30:31", time_delta=3 * u.s, n_samples=16)
    ts1["error_flag"] = 2 ** np.arange(16)
    with pytest.raises(ValueError):
        cal.parse_error_summary(ts1)
