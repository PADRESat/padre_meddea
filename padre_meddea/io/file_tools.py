"""
This module provides a generic file reader.
"""

from pathlib import Path
import datetime as dt

import numpy as np
import astropy.units as u
from astropy.timeseries import TimeSeries
from astropy.io import ascii

import ccsdspy
from ccsdspy import PacketField, PacketArray
from ccsdspy.utils import (
    count_packets,
    split_by_apid,
)

import padre_meddea

__all__ = ["read_file", "read_raw_file"]

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
    if filename.suffix == "bin":  # raw binary file
        result = read_raw_file(filename)
    elif filename.suffix == "fits":  # level 0 or above
        pass
    else:
        raise ValueError("File extension {filename.suffix} not recognized.")
    return result


def read_raw_file(filename: Path):
    """
    Read a level 0 data file.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    data : dict
        A dictionary of data arrays.
    """

    result = {
        "photons": parse_ph_packets(filename),
        "housekeeping": parse_hk_packets(filename),
        "spectra": parse_spectrum_packets(filename),
        "cmd_resp": parse_cmd_response_packets(filename),
    }

    return result


def parse_ph_packets(filename: Path):
    """Given a binary file, read only the photon packets and return an event list.

    Photon packets consist of 15 header words which includes a checksum.
    Each photon adds 3 words.

    Photon packet format is (words are 16 bits) described below.

        ==  =============================================================
        #   Description
        ==  =============================================================
        0   CCSDS header 1 (0x00A0)
        1   CCSDS header 2 (0b11 and sequence count)
        2   CCSDS header 3 payload size (remaining packet size - 1 octet)
        3   time_stamp_s 1
        4   time_stamp_s 2
        5   time_stamp_clocks 1
        6   time_stamp_clocks 2
        7   integration time in clock counts
        8   live time in clock counts
        9   drop counter ([15] Int.Time Overflow, [14:12] decimation level, [11:0] # dropped photons)
        10  checksum
        11  start of pixel data
        -   pixel time step in clock count
        -   pixel_location (ASIC # bits[7:5], pixel num bits[4:0])
        -   pixel_data 12 bit ADC count
        ==  =============================================================

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
    ph_data = pkt.load(packet_stream, include_primary_header=True)
    integration_time = ph_data["INTEGRATION_TIME"] * 12.8 * 1e-6
    live_time = ph_data["LIVE_TIME"] * 12.8 * 1e-6
    dead_time = (1.0 - live_time / integration_time) * 100
    packet_time = ph_data["TIME_S"] + ph_data["TIME_CLOCKS"] / 20.0e6
    total_hits = 0
    for this_packet in ph_data["PIXEL_DATA"]:
        total_hits += len(this_packet[0::3])

    hit_list = np.zeros((6, total_hits), dtype="uint16")
    time_stamps = np.zeros(total_hits)
    # 0 time, 1 asic_num, 2 channel num, 3 hit channel
    i = 0
    for this_pkt_num, this_ph_data, this_time in zip(
        ph_data["CCSDS_SEQUENCE_COUNT"],
        ph_data["PIXEL_DATA"],
        packet_time,
    ):
        num_hits = len(this_ph_data[0::3])
        ids = this_ph_data[1::3]
        asic_num = (ids & 0b11100000) >> 5
        channel_num = ids & 0b00011111
        hit_list[0, i : i + num_hits] = this_ph_data[0::3]
        time_stamps[i : i + num_hits] = this_ph_data[0::3] * 12.8e-6 + this_time
        hit_list[1, i : i + num_hits] = asic_num
        hit_list[2, i : i + num_hits] = channel_num
        hit_list[3, i : i + num_hits] = this_ph_data[2::3]
        hit_list[4, i : i + num_hits] = this_pkt_num
        hit_list[5, i : i + num_hits] = this_ph_data[0::3]
        i += num_hits

    ph_times = [EPOCH + dt.timedelta(seconds=this_t) for this_t in time_stamps]
    ph_list = TimeSeries(
        time=ph_times,
        data={
            "atod": hit_list[3, :],
            "asic": hit_list[1, :],
            "channel": hit_list[2, :],
            "clock": hit_list[5, :],
            "pktnum": hit_list[4, :],
            # "num": np.ones(len(hit_list[2, :])),
        },
        meta={
            "integration_times": integration_time,
            "live_times": live_time,
            "dead_times": dead_time,
            "total_hits": total_hits,
            "packet_time": packet_time,
            "time_s": ph_data["TIME_S"],
            "time_clocks": ph_data["TIME_CLOCKS"],
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
        A list of spectra data or None if no spectrum packets are found.
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


def parse_cmd_response_packets(filename: Path):
    """Given a raw binary file, read only the command response packets and return the data.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    cmd_resp_list : astropy.time.TimeSeries or None
        A list of command responses."""
    return None


def packet_definition_cmd_resp():
    """Return the packet definiton for a command response packet."""
    pass


def packet_definition_hk():
    """Return the packet definiton for the housekeeping packets."""
    hk_table = ascii.read(padre_meddea._data_directory / "hk_packet_def.csv")
    hk_table.add_index("name")
    p = [PacketField(name="TIMESTAMP", data_type="uint", bit_length=32)]
    for this_hk in hk_table["name"]:
        p += [PacketField(name=this_hk, data_type="uint", bit_length=16)]
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
