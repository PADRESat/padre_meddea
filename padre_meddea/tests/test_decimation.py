"""Test functions in calibration/decimation.py"""

import pytest
from pathlib import Path

import numpy as np
from astropy.time import Time


from padre_meddea.calibration.decimation import *
from padre_meddea import _data_directory

decimation_file_directory = _data_directory / "decimation"
decimation_file = decimation_file_directory / "20250314_decimation_table.csv"


@pytest.mark.parametrize(
    "time,result",
    [
        (Time("2025-04-01"), [3900, 11000]),
        (Time("2024-02-01T00:00"), [4000, 10000]),
        (Time("2025-03-13T01:00"), [3700, 20000]),
    ],
)
def test_get_decimation(time, result):
    """Check that we can read all files and see expected results"""
    assert np.all(get_decimation(time)[0] == result)


def test_read_decimation_file():
    """Check that we can read a specific file in depth"""
    result = read_decimation_file(decimation_file)
    assert np.all(result[0] == [3900, 11000])
    assert np.all(result[1] == [313, 482, 584, 755, 1094, 2113, 3813, 4096])
    decimation_array = result[2]
    assert np.all(decimation_array.shape == np.array([3, 8, 8]))
    assert np.all(decimation_array[1, 3, :] == [1.0, 1.0, 1.0, 1.0, 3.0, 9.0, 1.0, 1.0])
    assert np.all(
        decimation_array[2, 3, :] == [11.0, 11.0, 11.0, 11.0, 4.0, 1.0, 0.0, 24.0]
    )


def test_all_files():
    """Test that we can read every decimation file in the file table"""
    file_table = get_decimation_ftable()
    for this_row in file_table:
        result = read_decimation_file(
            _data_directory / "decimation" / this_row["filename"]
        )


def test_get_decimation_ftable():
    """Check features of the file table"""
    file_table = get_decimation_ftable()
    assert isinstance(file_table, TimeSeries)
    assert len(file_table) >= 1
    assert "filename" in file_table.keys()


@pytest.mark.parametrize(
    "time_str,filename",
    [
        (Time("2025-04-01T00:00"), "20250314_decimation_table.csv"),
        (Time("2025-03-13T01:00"), "20250313_decimation_table.csv"),
        (Time("2024-02-01T00:00"), "20240101_decimation_table.csv"),
        (Time("2022-02-01T00:00"), None),
    ],
)
def test_get_decimation_file(time_str, filename):
    """Check that expected results for specific times."""
    decim_filename = get_decimation_file(Time(time_str))
    if decim_filename:
        assert decim_filename.name == filename
    else:
        assert decim_filename is None
