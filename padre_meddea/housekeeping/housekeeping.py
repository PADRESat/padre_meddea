"""Module to provide functions for housekeeping data"""

import datetime
from pathlib import Path

import ccsdspy
import numpy as np
from astropy.io import ascii
from astropy.time import Time
from astropy.timeseries import TimeSeries
from ccsdspy import PacketField
from ccsdspy.utils import split_by_apid

from padre_meddea import APID, EPOCH, _data_directory, _package_directory, log

_data_directory = _package_directory / "data" / "housekeeping"
hk_definitions = ascii.read(_data_directory / "hk_packet_def.csv")
hk_definitions.add_index("name")


def parse_housekeeping_packets(filename: Path):
    """Given a raw file, read only the housekeeping packets and return a timeseries.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    hk_list : astropy.time.TimeSeries or None
        A list of housekeeping data
    """
    filename = Path(filename)
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_bytes = stream_by_apid.get(APID["housekeeping"], None)
    if packet_bytes is None:
        return None
    else:
        log.info(f"{filename.name}: Found housekeeping data")
    packet_definition = packet_definition_hk()
    pkt = ccsdspy.FixedLength(packet_definition)
    hk_data = pkt.load(packet_bytes, include_primary_header=True)
    hk_timestamps = [
        datetime.timedelta(seconds=int(this_t)) + EPOCH
        for this_t in hk_data["timestamp"]
    ]
    hk_data = TimeSeries(time=hk_timestamps, data=hk_data)
    hk_data.meta.update({"ORIGFILE": f"{filename.name}"})

    # Clean Housekeeping Times
    hk_data = clean_housekeeping_data(hk_data)

    return hk_data


def packet_definition_hk():
    """Return the packet definiton for the housekeeping packets."""
    p = [PacketField(name="timestamp", data_type="uint", bit_length=32)]
    for this_hk in hk_definitions["name"]:
        p += [PacketField(name=this_hk, data_type="uint", bit_length=16)]
    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]
    return p


def clean_housekeeping_data(hk_data: TimeSeries) -> TimeSeries:
    """
    Clean the housekeeping data by removing any bad timestamps.

    Parameters
    ----------
    hk_data : TimeSeries
        The housekeeping data to clean.

    Returns
    -------
    TimeSeries
        The cleaned housekeeping data.
    """
    # Calculates Differences in Time-Related Columns
    dts = hk_data.time[1:] - hk_data.time[:-1]
    timestamp_diff = hk_data["timestamp"][1:] - hk_data["timestamp"][:-1]

    # Calculate the Cadence for Interpolation
    median_dt = np.median(dts)
    median_timestamp_diff = np.median(timestamp_diff)

    bad_indices = np.argwhere(hk_data.time <= Time("2024-01-01T00:00"))
    for this_bad_index in bad_indices:
        if this_bad_index < len(hk_data.time) - 1:
            hk_data.time[this_bad_index] = hk_data.time[this_bad_index + 1] - median_dt
            hk_data["timestamp"][this_bad_index] = (
                hk_data["timestamp"][this_bad_index + 1] - median_timestamp_diff
            )
        else:
            hk_data.time[this_bad_index] = hk_data.time[this_bad_index - 1] + median_dt
            hk_data["timestamp"][this_bad_index] = (
                hk_data["timestamp"][this_bad_index - 1] + median_timestamp_diff
            )

    return hk_data
