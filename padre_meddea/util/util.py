"""
This module provides general utility functions.
"""

import os
from datetime import datetime, timezone
import time
from pathlib import Path
import warnings
import numpy as np


from astropy.time import Time, TimeDelta
import astropy.units as u
from ccsdspy.utils import split_packet_bytes, split_by_apid

from padre_meddea import EPOCH

__all__ = ["create_science_filename", "has_baseline"]

TIME_FORMAT = "%Y%m%dT%H%M%S"
VALID_DATA_LEVELS = ["l0", "l1", "ql", "l2", "l3", "l4"]
FILENAME_EXTENSION = ".fits"


def create_science_filename(
    time: str,
    level: str,
    version: str,
    mode: str = "",
    descriptor: str = "",
    test: bool = False,
):
    """Return a compliant filename. The format is defined as

    {mission}_{inst}_{mode}_{level}{test}_{descriptor}_{time}_v{version}.cdf

    This format is only appropriate for data level >= 1.

    Parameters
    ----------
    instrument : `str`
        The instrument name. Must be one of the following "eea", "nemesis", "merit", "spani"
    time : `str` (in isot format) or ~astropy.time
        The time
    level : `str`
        The data level. Must be one of the following "l0", "l1", "l2", "l3", "l4", "ql"
    version : `str`
        The file version which must be given as X.Y.Z
    descriptor : `str`
        An optional file descriptor.
    mode : `str`
        An optional instrument mode.
    test : bool
        Selects whether the file is a test file.

    Returns
    -------
    filename : `str`
        A CDF file name including the given parameters that matches the mission's file naming conventions

    Raises
    ------
    ValueError: If the instrument is not recognized as one of the mission's instruments
    ValueError: If the data level is not recognized as one of the mission's valid data levels
    ValueError: If the data version does not match the mission's data version formatting conventions
    ValueError: If the data product descriptor or instrument mode do not match the mission's formatting conventions
    """
    test_str = ""

    if isinstance(time, str):
        time_str = Time(time, format="isot").strftime(TIME_FORMAT)
    else:
        time_str = time.strftime(TIME_FORMAT)

    if level not in VALID_DATA_LEVELS[1:]:
        raise ValueError(
            f"Level, {level}, is not recognized. Must be one of {VALID_DATA_LEVELS[1:]}."
        )
    # check that version is in the right format with three parts
    if len(version.split(".")) != 3:
        raise ValueError(
            f"Version, {version}, is not formatted correctly. Should be X.Y.Z"
        )
    # check that version has integers in each part
    for item in version.split("."):
        try:
            int_value = int(item)
        except ValueError:
            raise ValueError(f"Version, {version}, is not all integers.")

    if test is True:
        test_str = "test"

    # the parse_science_filename function depends on _ not being present elsewhere
    if ("_" in mode) or ("_" in descriptor):
        raise ValueError(
            "The underscore symbol _ is not allowed in mode or descriptor."
        )

    filename = (
        f"padre_meddea_{mode}_{level}{test_str}_{descriptor}_{time_str}_v{version}"
    )
    filename = filename.replace("__", "_")  # reformat if mode or descriptor not given

    return filename + FILENAME_EXTENSION


def calc_time(pkt_time_s, pkt_time_clk, ph_clk=0):
    """
    Convert times to a Time object
    """
    deltat = TimeDelta(
        pkt_time_s * u.s + pkt_time_clk * 0.05 * u.us + ph_clk * 12.8 * u.us
    )
    result = Time(EPOCH + deltat)

    return result


def channel_to_pixel(channel: int) -> int:
    """
    Given a channel pixel number, return the pixel number.
    """
    CHANNEL_TO_PIX = {
        26: 0,
        15: 1,
        8: 2,
        1: 3,
        29: 4,
        13: 5,
        5: 6,
        0: 7,
        30: 8,
        21: 9,
        11: 10,
        3: 11,
        31: 12,
    }

    if channel in CHANNEL_TO_PIX.keys():
        return CHANNEL_TO_PIX[channel]
    else:
        warnings.warn(
            f"Found unconnected channel, {channel}. Returning channel + 12 ={channel+12}."
        )
        return channel + 12


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
        packet_stream = stream_by_apid[160]
        packet_bytes = split_packet_bytes(packet_stream)
        num_hits = np.zeros(packet_count)
        for i in range(packet_count):
            num_hits[i] = (len(packet_bytes[i]) - HEADER_BYTES) / BYTES_PER_PHOTON
    return np.sum(num_hits - np.floor(num_hits)) == 1
