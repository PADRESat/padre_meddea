"""
Provides general utility functions.
"""

import os
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
from ccsdspy.utils import split_by_apid, split_packet_bytes
from swxsoc.util import create_science_filename as swxsoc_create_science_filename
from swxsoc.util import parse_science_filename

import padre_meddea
from padre_meddea import APID, EPOCH, log

# used to identify bad times
MIN_TIME_BAD = Time("2024-02-01T00:00")

__all__ = [
    "parse_science_filename",
    "create_science_filename",
    "calc_time",
    "has_baseline",
    "is_consecutive",
]


def get_data_file_version() -> list:
    """Returns the correct version tuple to be used for data files."""
    # TODO: Andrew, uses padre_meddea version and uses csv file to figure out the data file version
    software_version_tuple = padre_meddea.__version__.split(".")
    software_version_tuple.reverse()
    return [1, 0, 0]


def create_science_filename(
    time: str,
    level: str,
    mode: str = "",
    descriptor: str = "",
    test: bool = False,
    overwrite=False,
) -> Path:
    """
    Return a compliant filename. It is a wrapper around `~swxsoc.util.create_science_filename`.
    The format is defined as

    padre_meddea_{mode}_{level}{test}_{descriptor}_{time}_v{version}.cdf

    This format is only appropriate for data level >= 0.
    The version number is automatically derived.

    Parameters
    ----------
    time : `str` (in isot format) or ~astropy.time
        The time
    level : `str`
        The data level. Must be one of the following "l0", "l1", "l2", "l3", "l4", "ql"
    descriptor : `str`
        An optional file descriptor.
    mode : `str`
        An optional instrument mode.
    test : bool
        Selects whether the file is a test file.
    overwrite : bool
        If False, then increments the z in x.y.z to the next available value

    Returns
    -------
    filename : `str`
        A file name
    """
    lambda_environment = os.getenv("LAMBDA_ENVIRONMENT")
    version_tuple = get_data_file_version()
    version_str = f"{version_tuple[0]}.{version_tuple[1]}.0"
    path = swxsoc_create_science_filename(
        "meddea",
        time=time,
        level=level,
        mode=mode,
        test=test,
        descriptor=descriptor,
        version=version_str,
    )
    if not overwrite:
        # now check if file exists, if so increment the z in the file version to max(z) + 1
        if lambda_environment:
            # TODO search for existing file in AWS for all files with x.y.z choose largest z and set to x.y.z+1
            # Andrew insert code here
            pass
        else:
            if Path(path).exists():
                search_pattern = path.replace(version_str, f"{version_str[0:-1]}*")
                existing_files = Path.cwd().glob(search_pattern)
                existing_versions = [
                    int(parse_science_filename(this_f)["version"][-1])
                    for this_f in existing_files
                ]
                path = swxsoc_create_science_filename(
                    "meddea",
                    time=time,
                    level=level,
                    descriptor=descriptor,
                    test=test,
                    version=f"{version_str[0:-1]}{max(existing_versions) + 1}",
                )
    if lambda_environment:
        temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
        path = temp_dir / path
    return Path(path)


def calc_time(pkt_time_s, pkt_time_clk=0, ph_clk=0) -> Time:
    """
    Convert times to a Time object
    """
    deltat = TimeDelta(
        pkt_time_s * u.s
        + pkt_time_clk * 0.05 * u.microsecond
        + ph_clk * 12.8 * u.microsecond
    )
    result = Time(EPOCH + deltat)
    return result


def has_baseline(filename: Path, packet_count=10) -> bool:
    """Given a stream of photon packets, check whether the baseline measurement is included.
    Baseline packets have one extra word per photon for a total of 4 words (8 bytes).

    This function calculates the number of hits in the packet assuming 4 words per photon.
    If the resultant is not an integer number then returns False.

    Parameters
    ----------
    packet_bytes : byte string
        Photon packet bytes, must be an integer number of whole packets and greaterh

    Returns
    -------
    result : bool

    """
    HEADER_BYTES = 11 * 16 / 8
    BYTES_PER_PHOTON = 16 * 4 / 8

    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
        if APID["photon"] in stream_by_apid.keys():  # only applicable to photon packets
            packet_stream = stream_by_apid[APID["photon"]]
            packet_bytes = split_packet_bytes(packet_stream)
            packet_count = min(
                len(packet_bytes), packet_count
            )  # in case we have fewer than packet_count in the file
            num_hits = np.zeros(packet_count)
            for i in range(packet_count):
                num_hits[i] = (len(packet_bytes[i]) - HEADER_BYTES) / BYTES_PER_PHOTON
        else:
            raise ValueError("Only works on photon packets.")
    # check if there is any remainder for non integer number of hits
    return np.sum(num_hits - np.floor(num_hits)) == 0


def str_to_fits_keyword(keyword: str) -> str:
    """Given a keyword string, return a fits compatible keyword string
    which must not include special characters and have fewer have no more
    than 8 characters."""
    clean_keyword = "".join(e for e in keyword if e.isalnum()).strip().upper()
    return clean_keyword[0:8]


def is_consecutive(arr: np.array) -> bool:
    """Return True if the packet sequence numbers are all consecutive integers, has no missing numbers."""
    MAX_SEQCOUNT = 2**14 - 1  # 16383

    # Ensure arr is at least 1D
    arr = np.atleast_1d(arr)

    # check if seqcount has wrapped around
    indices = np.where(arr == MAX_SEQCOUNT)
    if len(indices[0]) == 0:  # no wrap
        return np.all(np.diff(arr) == 1)
    else:
        last_index = 0
        result = True
        for this_ind in indices[0]:
            this_arr = arr[last_index : this_ind + 1]
            result = result & np.all(np.diff(this_arr) == 1)
            last_index = this_ind + 1
        # now do the remaining part of the array
        this_arr = arr[last_index + 1 :]
        result = result & np.all(np.diff(this_arr) == 1)
        return result


def trim_timeseries(ts: TimeSeries, t0=MIN_TIME_BAD) -> TimeSeries:
    """Remove all times in a time series before a given time.
    Parameters
    ----------
    ts : Time
        The time before which all data should be removed.

    Returns
    -------
    TimeSeries
        A TimeSeries containing the trimmed time series.
    """
    inds = ts.time < t0
    bad_count = np.sum(inds)
    if bad_count > 0:
        log.warning(f"Found {bad_count} bad times. Removing them.")
    return ts[~inds]


def parse_ph_flags(ph_flags):
    """Given the photon flag field, parse into its individual components.
    The flags are stored as follows
    [15] Int.Time Overflow, [14:12] decimation level, [11:0] # dropped photons

    Returns
    -------
    decim_lvl, dropped_count, int_time_overflow
    """
    decim_lvl = (ph_flags >> 12) & 0b0111
    dropped_count = ph_flags & 2047
    int_time_overflow = ph_flags & 32768
    return decim_lvl, dropped_count, int_time_overflow


def threshold_to_energy(threshold_value: int) -> u.Quantity:
    """Given a threshold value return the threshold energy.
    If given a threshold of 63, the pixel is disabled.
        In that case, return the maximum detectable energy of 100 keV
    Parameters
    ----------
    threshold_value : int
        The threshold value
    Returns
    -------
    threshold_energy : u.Quantity
        The energy value of the threshold.
    """
    MAX_ENERGY = 100 * u.keV
    max_threshold = 63
    if (0 > threshold_value) or (threshold_value > max_threshold):
        raise ValueError("Threshold value should be between 0 and 63.")
    if threshold_value == 63:
        return MAX_ENERGY
    msb = 0.8 * u.keV
    lsb = 0.2 * u.keV
    lsb_array = np.arange(-3, 60, 1)
    lsb_array[lsb_array > 52] = 52
    msb_array = np.zeros(max_threshold)
    msb_array[-8:] = np.arange(0, 8)
    threshold_energy = msb_array * msb + lsb_array * lsb
    return threshold_energy[threshold_value]


def get_file_time(filename) -> Time:
    """Given filename return the time stamp."""
    return Time(f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}T00:00")
