import csv
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries

import padre_meddea
import padre_meddea.util.util as util
from padre_meddea import EPOCH

TIME = "2024-04-06T12:06:21"
TIME_FORMATTED = "20240406T120621"


@pytest.fixture
def mock_version_mapping(tmpdir):
    """Create a temporary version mapping CSV file for testing."""
    # Create a test version mapping file
    mapping_file = tmpdir / "software_to_data_version_mapping.csv"

    # Write test data to the file
    with open(mapping_file, "w", newline="") as f:
        writer = csv.writer(f)
        # Format: software_version, data_version
        writer.writerow(["0.1", "0.1"])
        writer.writerow(["1.0", "1.0"])
        writer.writerow(["1.0", "1.1"])  # Intentionally out of order to test sorting
        writer.writerow(["1.1", "1.2"])
        writer.writerow(["2.0", "2.0"])
        writer.writerow([])  # Empty row to test empty row handling

    return mapping_file


@pytest.mark.parametrize(
    "current_version,expected_data_version",
    [
        ("0.1.5", "0.1"),  # Exact match for major.minor
        (
            "1.0.3",
            "1.1",
        ),  # Match with multiple data versions (should get lowest after sorting)
        ("1.1.2", "1.2"),  # Another exact match
        ("2.0.0", "2.0"),  # Exact match for latest version
        ("1.5.0", "2.0"),  # No match, should default to latest version
        ("3.0.0", "2.0"),  # No match, should default to latest version
    ],
)
def test_get_filename_version_base(
    mock_version_mapping, current_version, expected_data_version, monkeypatch
):
    """Test the get_filename_version_base function with various software versions."""
    # Patch the _data_directory to point to our temporary directory
    monkeypatch.setattr(
        padre_meddea, "_data_directory", Path(mock_version_mapping).parent
    )

    # Patch the software version
    monkeypatch.setattr(padre_meddea, "__version__", current_version)

    # Call the function and check the result
    assert util.get_filename_version_base() == expected_data_version


# fmt: off
@pytest.mark.parametrize(
    "level,data_type,input,output,version_index",
    [
        ("l0", "housekeeping", "1.2.3", "1.2.4", 0),
        ("l0", "spectrum", "1.0.6", "1.0.7", 0),
        ("l0", "spectrum", "1.0.6", "1.1.6", 1),
        ("l0", "photon", "1.0.6", "2.0.6", 2)
    ],
)
def test_increment_filename_version(level, data_type, input, output, version_index):
    fname = util.create_science_filename('meddea', time=Time(TIME),
                level=level,
                descriptor=data_type,
                test=False,
                version=input)
    new_fname = util.increment_filename_version(fname, version_index=version_index)
    tokens = util.parse_science_filename(new_fname)
    assert tokens['version'] == output


# fmt: off
@pytest.mark.parametrize("instrument,time,level,version,result", [
    ("meddea", TIME, "l1", "1.2.3", f"padre_meddea_l1_{TIME_FORMATTED}_v1.2.3.fits"),
    ("meddea", TIME, "l2", "2.4.5", f"padre_meddea_l2_{TIME_FORMATTED}_v2.4.5.fits"),
    ("sharp", TIME, "l2", "1.3.5", f"padre_sharp_l2_{TIME_FORMATTED}_v1.3.5.fits"),
    ("sharp", TIME, "l3", "2.4.5", f"padre_sharp_l3_{TIME_FORMATTED}_v2.4.5.fits"),
]
)
def test_science_filename_output_a(instrument, time, level, version, result):
    """Test simple cases with expected output.
    Since we are using the swxsoc create_science_filename, we are testing whether we did the config correctly in __init__.py"""
    assert (
        util.create_science_filename(instrument, time, level=level, version=version)
        == result
    )
# fmt: on


def test_is_consecutive():
    """Test if consecutive"""
    assert util.is_consecutive(np.arange(10))
    assert util.is_consecutive(range(100))
    assert util.is_consecutive(np.arange(10, 100, 1))
    # test if the array loops over
    assert util.is_consecutive(
        np.concatenate((np.arange(0, 2**14), np.arange(0, 2000)))
    )


def test_is_not_consecutive():
    """Test if not consecutive"""
    assert not util.is_consecutive([0, 2, 3, 4, 5])
    assert not util.is_consecutive(
        np.concatenate((np.arange(1, 10), np.arange(11, 20)))
    )


def test_has_baseline():
    assert not util.has_baseline(
        padre_meddea._test_files_directory / "apid160_4packets.bin"
    )


def test_has_baseline_error():
    with pytest.raises(ValueError):
        util.has_baseline(padre_meddea._test_files_directory / "apid162_4packets.bin")


@pytest.mark.parametrize(
    "pkt_time_s,pkt_time_clk,ph_clk,output",
    [
        (0, 0, 0, EPOCH),
        (1, 0, 0, EPOCH + TimeDelta(1 * u.s)),
        (10, 0, 0, EPOCH + TimeDelta(10 * u.s)),
        (0, 1, 0, EPOCH + TimeDelta(0.05 * u.microsecond)),
        (0, 0, 1, EPOCH + TimeDelta(12.8 * u.microsecond)),
        (
            5,
            5,
            5,
            EPOCH
            + TimeDelta(5 * u.s + 5 * 0.05 * u.microsecond + 5 * 12.8 * u.microsecond),
        ),
    ],
)
def test_calc_time(pkt_time_s, pkt_time_clk, ph_clk, output):
    assert util.calc_time(pkt_time_s, pkt_time_clk, ph_clk) == output


def test_threshold_to_energy_error():
    with pytest.raises(ValueError):
        util.threshold_to_energy(-1)
    with pytest.raises(ValueError):
        util.threshold_to_energy(64)
    with pytest.raises(ValueError):
        util.threshold_to_energy(100)


@pytest.mark.parametrize(
    "input,output",
    [
        (0, -0.6 * u.keV),
        (55, 10.4 * u.keV),
        (62, 16 * u.keV),
    ],
)
def test_threshold_to_energy(input, output):
    assert_quantity_allclose([util.threshold_to_energy(input)], [output])


def test_trim_timeseries():
    # all bad
    ts = TimeSeries(
        time_start=(util.MIN_TIME_BAD - 10 * u.year), time_delta=1 * u.year, n_samples=5
    )
    assert len(util.trim_timeseries(ts)) == 0
    # some bad
    ts = TimeSeries(
        time_start=(util.MIN_TIME_BAD - 1 * u.year), time_delta=1 * u.year, n_samples=5
    )
    assert len(util.trim_timeseries(ts)) == len(ts) - 1
    # all good
    ts = TimeSeries(time_start=util.MIN_TIME_BAD, time_delta=1 * u.year, n_samples=5)
    assert len(util.trim_timeseries(ts)) == len(ts)
