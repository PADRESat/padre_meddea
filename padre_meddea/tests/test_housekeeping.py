import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time
from astropy.timeseries import TimeSeries

import padre_meddea.housekeeping.calibration as cal
from padre_meddea import _test_files_directory
from padre_meddea.housekeeping import housekeeping as hk

hk_packet_file = _test_files_directory / "padreMDU8_240916122904.dat"

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


def test_calibrate_spaceops_hk_values():
    data = {
        "HVTemp": 46720,
        "HVCurrent": 65472,
        "sysError": 160,
        "phRate": 22,
        "goodCmdCount": 149,
        "TimeStamp1": 12680,
        "Amps_3V3_D": 36876,
        "TimeStamp2": 54499,
        "Amps_3V3_A": 37084,
        "SequenceCount": 61656,
        "DIBTemp": 55824,
        "Amps_1V5": 38808,
        "decimationRate": 0,
        "HVVolts": 29759,
        "FPTemp": 34865,
        "errorCount": 0,
        "heaterPWM": 24077,
    }
    calibrated_data = cal.calibrate_spaceops_hk_values(data)
    expected_calibrated_data = {
        "HVTemp": -15.47143951 * u.deg_C,
        "HVCurrent": 11.06977848 * u.nA,
        "sysError": 160,
        "phRate": 22,
        "goodCmdCount": 149,
        "TimeStamp1": 12680,
        "Amps_3V3_D": 48.794224 * u.mA,
        "TimeStamp2": 54499,
        "Amps_3V3_A": 46.552816 * u.mA,
        "SequenceCount": 61656,
        "DIBTemp": -8.32003798 * u.deg_C,
        "Amps_1V5": 27.974992 * u.mA,
        "decimationRate": 0,
        "HVVolts": -299.2749 * u.V,
        "FPTemp": -19.90109367 * u.deg_C,
        "errorCount": 0,
        "heaterPWM": 38.5232 * u.percent,
        "time": Time("2026-05-02T15:29:34.000", scale="utc"),
    }
    for key in expected_calibrated_data:
        if isinstance(expected_calibrated_data[key], u.Quantity):
            assert_quantity_allclose(
                calibrated_data[key], expected_calibrated_data[key]
            )
        else:
            assert calibrated_data[key] == expected_calibrated_data[key]
