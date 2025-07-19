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
from padre_meddea.housekeeping.register import add_register_address_name
from padre_meddea.util.util import calc_time, MIN_TIME_BAD

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
    hk_pkttimess = [
        datetime.timedelta(seconds=int(this_t)) + EPOCH
        for this_t in hk_data["pkttimes"]
    ]
    hk_data = TimeSeries(time=hk_pkttimess, data=hk_data)
    hk_data.meta.update({"ORIGFILE": f"{filename.name}"})

    # Clean Housekeeping Times
    #hk_data = clean_housekeeping_data(hk_data)

    return hk_data


def packet_definition_hk():
    """Return the packet definiton for the housekeeping packets."""
    p = [PacketField(name="pkttimes", data_type="uint", bit_length=32)]
    for this_hk in hk_definitions["name"]:
        p += [PacketField(name=this_hk, data_type="uint", bit_length=16)]
    p += [PacketField(name="checksum", data_type="uint", bit_length=16)]
    return p



def parse_cmd_response_packets(filename: Path):
    """Given a raw binary file, read only the command response packets and return the data.

    The packet is defined as follows

    ==  =============================================================
    #   Description
    ==  =============================================================
    0   CCSDS header 1 (0x00A5)
    1   CCSDS header 2 (0b11 and sequence count)
    2   CCSDS header 3 payload size (remaining packet size - 1 octet)
    3   time_stamp_s 1
    4   time_stamp_s 2
    5   time_stamp_clocks 1
    6   time_stamp_clocks 2
    7   register address
    8   register value
    9   checksum
    ==  =============================================================

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    cmd_resp_list : astropy.time.TimeSeries or None
        A list of register read responses.
    """
    filename = Path(filename)
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_bytes = stream_by_apid.get(APID["cmd_resp"], None)
    if packet_bytes is None:
        return None
    else:
        log.info(f"{filename.name}: Found read data")
    packet_definition = packet_definition_cmd_response()
    pkt = ccsdspy.FixedLength(packet_definition)
    data = pkt.load(packet_bytes, include_primary_header=True)
    timestamps = calc_time(data["pkttimes"], data["pktclock"])
    data = {
        "pkttimes": data["pkttimes"],
        "pktclock": data["pktclock"],
        "address": data["address"],
        "value": data["value"],
        "seqcount": data["CCSDS_SEQUENCE_COUNT"],
    }
    ts = TimeSeries(time=timestamps, data=data)
    ts = add_register_address_name(ts)
    ts.meta.update({"ORIGFILE": f"{filename.name}"})

    # Clean Command Response Times
    #ts = clean_cmd_response_data(ts)

    return ts


def packet_definition_cmd_response():
    """Return the packet definiton for the register read response"""
    p = [
        PacketField(name="pkttimes", data_type="uint", bit_length=32),
        PacketField(name="pktclock", data_type="uint", bit_length=32),
        PacketField(name="address", data_type="uint", bit_length=16),
        PacketField(name="value", data_type="uint", bit_length=16),
        PacketField(name="checksum", data_type="uint", bit_length=16),
    ]
    return p


def clean_hk_data(hk_ts: TimeSeries) -> TimeSeries:
    """
    Clean the housekeeping data by replacing any bad times with interpolated times.

    Parameters
    ----------
    hk_ts : TimeSeries
        The housekeeping data to clean.

    Returns
    -------
    TimeSeries
        The cleaned housekeeping data.
    """
    # Calculates Differences in Time-Related Columns
    dts = hk_ts.time[1:] - hk_ts.time[:-1]
    pkttimes_diff = hk_ts["pkttimes"][1:] - hk_ts["pkttimes"][:-1]

    # Calculate the Cadence for Interpolation
    median_dt = np.median(dts)
    median_pkttimes_diff = np.median(pkttimes_diff)

    bad_indices = np.argwhere(hk_ts.time < MIN_TIME_BAD)
    for this_bad_index in bad_indices:
        if this_bad_index < len(hk_ts.time) - 1:
            hk_ts.time[this_bad_index] = hk_ts.time[this_bad_index + 1] - median_dt
            hk_ts["pkttimes"][this_bad_index] = (
                hk_ts["pkttimes"][this_bad_index + 1] - median_pkttimes_diff
            )
        else:
            hk_ts.time[this_bad_index] = hk_ts.time[this_bad_index - 1] + median_dt
            hk_ts["pkttimes"][this_bad_index] = (
                hk_ts["pkttimes"][this_bad_index - 1] + median_pkttimes_diff
            )

    return hk_ts
