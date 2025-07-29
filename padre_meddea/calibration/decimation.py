"""Utilities to deal with decimation"""

from glob import glob
from pathlib import Path

import numpy as np

from astropy.io import ascii
from astropy.time import Time
from astropy.timeseries import TimeSeries

from padre_meddea import _data_directory


def get_decimation(this_time: Time):
    """Given a time, return the decimation parameters that was in effect at the time"""
    filename = get_decimation_file(this_time)
    return read_decimation_file(_data_directory / "decimation" / filename)


def get_decimation_file(this_time: Time) -> Path:
    """Given a time, return the decimation filename that was in effect at the time"""
    file_table = get_decimation_ftable()
    ind = file_table.time <= this_time
    if np.any(ind):
        return Path(_data_directory / "decimation" / file_table[ind][-1]["filename"])
    else:
        return Path("")


def get_decimation_ftable():
    """Return a time series table of decimation files

    Returns
    -------
    file time series
    """
    decimation_file_directory = _data_directory / "decimation"
    file_list = glob(str(decimation_file_directory) + "/*.csv")
    times = []
    filenames = []
    for this_file in file_list:
        fname = Path(this_file).name
        filenames.append(fname)
        times.append(Time(f"{fname[0:4]}-{fname[4:6]}-{fname[6:8]}T00:00"))
    result = TimeSeries(time=times, data={"filename": filenames})
    result.sort('time')
    return result


def read_decimation_file(file_path: Path):
    """Read and parse a decimation file.

    Parameters
    ----------
    file_path : Path

    Returns
    -------
    [up, down], edges, decimation_correction_array
    """
    raw_data = ascii.read(file_path)
    up = raw_data[0][2]
    down = raw_data[0][3]
    edges = raw_data[2][2:]
    num_edges = len(edges)
    num_levels = 8

    decimation_correction = np.zeros((3, num_edges, num_levels))
    # TODO: there must be a better way to cast an astropy table to a numpy array...
    for this_level in range(num_levels):
        for this_edge in range(num_edges):
            keeps = raw_data[3 + this_level][this_edge + 2]
            discards = raw_data[3 + this_level + num_levels][this_edge + 2]
            decimation_correction[1, this_level, this_edge] = keeps
            decimation_correction[2, this_level, this_edge] = discards
            decimation_correction[0, this_level, this_edge] = (keeps + discards) / keeps
    return np.array([up, down]), np.array(edges), decimation_correction
