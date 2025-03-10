"""
This module provides general utility functions.
"""

from pathlib import Path
import warnings
import numpy as np

from astropy.time import Time, TimeDelta
import astropy.units as u
from ccsdspy.utils import split_packet_bytes, split_by_apid

from swxsoc.util import create_science_filename

from padre_meddea import EPOCH, APID

__all__ = [
    "create_science_filename",
    "calc_time",
    "has_baseline",
    "is_consecutive",
    "channel_to_pixel",
]


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
        18: 5,
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


def pixel_to_str(pixel_num: int) -> str:
    """
    Given a pixel number, return a standardized string.
    """
    if not (0 <= pixel_num <= 11):
        raise ValueError("Pixel integer number must be 0 to 11.")
    if 0 <= pixel_num <= 7:
        pixel_size = "L"
    elif 8 <= pixel_num <= 12:
        pixel_size = "S"
    return f"Pixel{pixel_num}{pixel_size}"


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
