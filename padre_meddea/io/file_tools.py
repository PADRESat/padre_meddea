"""
This module provides a generic file reader.
"""

from pathlib import Path
import datetime as dt

import numpy as np
import astropy.units as u
from astropy.timeseries import TimeSeries

import ccsdspy
from ccsdspy import PacketField, PacketArray
from ccsdspy.utils import (
    count_packets,
    split_by_apid,
)

__all__ = ["read_file"]

APID = {
    "spectrum": 0xA2,  # decimal 162
    "photon": 0xA0,  # decimal 160
    "housekeeping": 0xA3,  # decimal 163
    "cmd_resp": 0xA5,  # decimal 165
}
EPOCH = dt.datetime(2000, 1, 1)


def read_file(filename: Path):
    """
    Read a file.

    Parameters
    ----------
    filename: Path
        A file to read.

    Returns
    -------
    data

    Examples
    --------
    """
    result = read_l0_file(filename)
    return result


def read_l0_file(filename: Path, include_ccsds_headers: bool = True):
    """
    Read a level 0 data file.

    Parameters
    ----------
    filename : Path
        A file to read
    include_ccsds_headers : bool
        If True then return the CCSDS headers in the data arrays.

    Returns
    -------
    data : dict
        A dictionary of data arrays.
    """
    
    result = {"photons": parse_ph_packets(filename),
              "housekeeping": parse_hk_packets(filename),
              "spectra": parse_spectrum_packets(filename)}

    return result


def parse_ph_packets(filename: Path):
    """Given a binary file, read only the photon packets and return a photon list.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    ph_list : astropy.time.TimeSeries or None
        A photon list
    """
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_stream = stream_by_apid.get(APID["photon"], None)
    if packet_stream is None:
        return None
    packet_definition = packet_definition_ph()
    pkt = ccsdspy.VariableLength(packet_definition)
    ph_data = pkt.load(packet_stream)

    integration_time = ph_data["INTEGRATION_TIME"] * 12.8 * 1e-6
    live_time = ph_data["LIVE_TIME"] * 12.8 * 1e-6
    dead_time = (1.0 - live_time / integration_time) * 100
    total_hits = 0
    for this_packet in ph_data["PIXEL_DATA"]:
        total_hits += len(this_packet[0::3])

    hit_list = np.zeros((4, total_hits), dtype="uint32")
    time_stamps = np.zeros(total_hits)
    # 0 time, 1 asic_num, 2 channel num, 3 hit channel
    i = 0
    for this_ph_data, this_s, this_clock in zip(
        ph_data["PIXEL_DATA"], ph_data["TIME_S"], ph_data["TIME_CLOCKS"]
    ):
        packet_time = this_s + this_clock / 20.0e6
        num_hits = len(this_ph_data[0::3])
        ids = this_ph_data[1::3]
        asic_num = (ids & 0b11100000) >> 5
        channel_num = ids & 0b00011111
        hit_list[0, i : i + num_hits] = this_ph_data[0::3]
        time_stamps[i : i + num_hits] = this_ph_data[0::3] * 12.8e-6 + packet_time
        hit_list[1, i : i + num_hits] = asic_num
        hit_list[2, i : i + num_hits] = channel_num
        hit_list[3, i : i + num_hits] = this_ph_data[2::3]
        i += num_hits

    ph_times = [EPOCH + dt.timedelta(seconds=this_t) for this_t in time_stamps]
    ph_list = TimeSeries(
        time=ph_times,
        data={
            "energy": hit_list[3, :],
            "channel": hit_list[2, :],
            "num": np.ones(len(hit_list[2, :])),
        },
        meta={
            "integration_times": integration_time,
            "live_times": live_time,
            "dead_times": dead_time,
            "total_hits": total_hits,
        },
    )
    ph_list.sort()
    int_time = (ph_list.time.max() - ph_list.time.min()).to("min")
    ph_list.meta.update({"int_time": int_time})
    ph_list.meta.update({"avg rate": (total_hits * u.ct) / int_time})
    return ph_list


def parse_hk_packets(filename: Path):
    """Given a binary file, read only the housekeeping packets and return a timeseries.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    hk_list : astropy.time.TimeSeries or None
        A list of housekeeping data
    """
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_stream = stream_by_apid.get(APID["housekeeping"], None)
    if packet_stream is None:
        return None
    packet_definition = packet_definition_hk()
    pkt = ccsdspy.FixedLength(packet_definition)
    hk_data = pkt.load(packet_stream)
    hk_timestamps = [
        dt.timedelta(seconds=int(this_t)) + EPOCH for this_t in hk_data["TIMESTAMP"]
    ]
    hk_data = TimeSeries(time=hk_timestamps, data=hk_data)
    return hk_data


def parse_spectrum_packets(filename: Path):
    """Given a binary file, read only the spectrum packets and return the data.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    hk_list : astropy.time.TimeSeries or None
        A list of spectra data
    """
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_stream = stream_by_apid.get(APID["spectrum"], None)
    if packet_stream is None:
        return None
    packet_definition = packet_definition_hist2()
    pkt = ccsdspy.FixedLength(packet_definition)
    data = pkt.load(packet_stream)
    timestamps = [
        dt.timedelta(seconds=int(this_t)) + EPOCH for this_t in data["TIMESTAMPS"]
    ]
    num_packets = len(timestamps)
    h = data["HISTOGRAM_DATA"].reshape((num_packets, 24, 513))
    histogram_data = h[:, :, 1:]  # remove the id field
    ids = h[:, :, 0]
    return timestamps, histogram_data, ids


def packet_definition_hk():
    """Return the packet definiton for the housekeeping packets."""
    NUM_FIELDS = 8
    p = [PacketField(name="TIMESTAMP", data_type="uint", bit_length=32)]
    for i in range(NUM_FIELDS):
        p += [PacketField(name=f"HK{i}", data_type="uint", bit_length=16)]
    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]
    return p


def packet_definition_hist():
    """Return the packet definition for the histogram packets."""
    # NOTE: This is an outdated packet definition.
    # the number of pixels provided by a histogram packet
    NUM_BINS = 512
    NUM_PIXELS = 24

    # the header
    p = [
        PacketField(name="TIMESTAMPS", data_type="uint", bit_length=4 * 16),
        PacketField(name="TIMESTAMPCLOCK", data_type="uint", bit_length=4 * 16),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=32),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=32),
    ]

    for i in range(NUM_PIXELS):
        p += [
            PacketField(name=f"HISTOGRAM_SYNC{i}", data_type="uint", bit_length=8),
            PacketField(name=f"HISTOGRAM_DETNUM{i}", data_type="uint", bit_length=3),
            PacketField(name=f"HISTOGRAM_PIXNUM{i}", data_type="uint", bit_length=5),
            PacketArray(
                name=f"HISTOGRAM_DATA{i}",
                data_type="uint",
                bit_length=16,
                array_shape=NUM_BINS,
            ),
        ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_hist2():
    """Return the packet definition for the histogram packets."""
    # the number of pixels provided by a histogram packet
    NUM_BINS = 512
    NUM_PIXELS = 24

    # the header
    p = [
        PacketField(name="TIMESTAMPS", data_type="uint", bit_length=32),
        PacketField(name="TIMESTAMPCLOCK", data_type="uint", bit_length=32),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=32),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=32),
    ]

    p += [
        PacketArray(
            name="HISTOGRAM_DATA",
            data_type="uint",
            bit_length=16,
            array_shape=NUM_PIXELS * (NUM_BINS + 1),
        ),
    ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_ph():
    """Return the packet definition for the photon packets."""
    p = [
        PacketField(name="TIME_S", data_type="uint", bit_length=32),
        PacketField(name="TIME_CLOCKS", data_type="uint", bit_length=32),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=16),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=16),
        PacketField(name="FLAGS", data_type="uint", bit_length=16),
        PacketField(name="CHECKSUM", data_type="uint", bit_length=16),
        PacketArray(
            name="PIXEL_DATA", data_type="uint", bit_length=16, array_shape="expand"
        ),
    ]
    return p


def inspect_raw_file(filename: Path):
    """Given a raw binary file of packets, provide some high level summary information."""
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)

    num_packets = count_packets(filename)
    print(f"There are {num_packets} total packets in this file")
    # TODO move this message to the logger, also assumes that there are no unknown APIDs
    print(f"APIDs found {stream_by_apid.keys()}.")

    for key, val in APID.items():
        if val in stream_by_apid.keys():
            print(
                f"There are {count_packets(stream_by_apid[val])} {key} packets (APID {val})."
            )
