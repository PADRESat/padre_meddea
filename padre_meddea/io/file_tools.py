"""
This module provides a generic file reader.
"""

from pathlib import Path
import datetime as dt

import numpy as np
import astropy.units as u
from astropy.timeseries import TimeSeries
from astropy.io import ascii
from astropy.table import Table
from astropy.time import Time

import ccsdspy
from ccsdspy import PacketField, PacketArray
from ccsdspy.utils import (
    count_packets,
    split_by_apid,
)
import astropy.io.fits as fits

import padre_meddea
from padre_meddea import EPOCH, APID
from padre_meddea.util.util import has_baseline, calc_time, channel_to_pixel

__all__ = ["read_file", "read_raw_file", "read_fits"]


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
        result = read_raw_file(Path(filename))
    elif filename.suffix == "fits":  # level 0 or above
        read_fits(Path(filename))
    else:
        raise ValueError("File extension {filename.suffix} not recognized.")
    return result


def read_raw_file(filename: Path):
    """
    Read a level 0 data file.

    Parameters
    ---------
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


def read_fits(filename: Path):
    """
    Read a fits file.
    """
    hdu = fits.open(filename)

    if (hdu[0].header["LEVEL"] == 0) and (hdu[0].header["DATATYPE"] == "event_list"):
        event_list = read_fits_l0_event_list(
            filename
        )  # do I need to close the file since it is being opened again right after this?
        return event_list
    if (hdu[0].header["LEVEL"] == 0) and (hdu[0].header["DATATYPE"] == "housekeeping"):
        hk_list = read_fits_l0_housekeeping(filename)
        return hk_list
    else:
        raise ValueError(f"File contents of {filename} not recogized.")


def read_fits_l0_event_list(filename: Path) -> TimeSeries:
    """ """
    hdu = fits.open(filename)
    num_events = len(hdu["SCI"].data["seqcount"])
    ph_times = calc_time(
        hdu["sci"].data["pkttimes"],
        hdu["sci"].data["pktclock"],
        hdu["sci"].data["clocks"],
    )
    # add the pixel conversions
    pixel = np.array(
        [channel_to_pixel(this_chan) for this_chan in hdu["sci"].data["channel"]],
        dtype=np.uint8,
    )
    event_list = TimeSeries(
        time=ph_times,
        data={
            "atod": hdu["sci"].data["atod"],
            "asic": hdu["sci"].data["asic"],
            "channel": hdu["sci"].data["channel"],
            "pixel": pixel,
            "clocks": hdu["sci"].data["clocks"],
            "pktnum": hdu["sci"].data["seqcount"],
        },
    )
    event_list.sort()
    return event_list


def read_fits_l0_housekeeping(filename: Path) -> TimeSeries:
    """Read a level 0 housekeeping file"""
    hdu = fits.open(filename)
    colnames = [this_col.name for this_col in hdu["HK"].data.columns]
    times = calc_time(hdu["HK"].data["timestamp"])
    hk_list = TimeSeries(
        time=times, data={key: hdu["hk"].data[key] for key in colnames}
    )
    return hk_list


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
    print(packet_stream)
    if packet_stream is None:
        return None
    packet_definition = packet_definition_ph()
    pkt = ccsdspy.VariableLength(packet_definition)
    ph_data = pkt.load(packet_stream, include_primary_header=True)
    # packet_time = ph_data["TIME_S"] + ph_data["TIME_CLOCKS"] / 20.0e6
    if has_baseline(filename):
        WORDS_PER_HIT = 4
    else:
        WORDS_PER_HIT = 3

    pkt_list = Table()
    pkt_list["seqcount"] = ph_data["CCSDS_SEQUENCE_COUNT"]
    pkt_list["pkttimes"] = ph_data["TIME_S"]
    pkt_list["pktclock"] = ph_data["TIME_CLOCKS"]
    pkt_list["livetime"] = ph_data["LIVE_TIME"]
    pkt_list["inttime"] = ph_data["INTEGRATION_TIME"]
    pkt_list["flags"] = ph_data["FLAGS"]

    total_hits = 0
    for this_packet in ph_data["PIXEL_DATA"]:
        total_hits += len(this_packet[0::WORDS_PER_HIT])

    if (total_hits - np.floor(total_hits)) != 0:
        raise ValueError(
            f"Got non-integer number of hits {total_hits - np.floor(total_hits)}."
        )

    # hit_list definition
    # 0, raw id number
    # 1, asic number, 0 to 7
    # 2, channel number, 0 to 32
    # 3, energy 12 bits
    # 4, packet sequence number 12 bits
    # 5, clock number
    # 6, baseline (if exists) otherwise 0

    hit_list = np.zeros((9, total_hits), dtype="uint16")
    time_s = np.zeros(total_hits, dtype="uint32")
    time_clk = np.zeros(total_hits, dtype="uint32")
    # 0 time, 1 asic_num, 2 channel num, 3 hit channel
    i = 0
    # iterate over packets
    for this_pkt_num, this_ph_data, pkt_s, pkt_clk in zip(
        ph_data["CCSDS_SEQUENCE_COUNT"],
        ph_data["PIXEL_DATA"],
        ph_data["TIME_S"],
        ph_data["TIME_CLOCKS"],
    ):
        num_hits = len(this_ph_data[0::WORDS_PER_HIT])
        ids = this_ph_data[1::WORDS_PER_HIT]
        asic_num = (ids & 0b11100000) >> 5
        channel_num = ids & 0b00011111
        hit_list[0, i : i + num_hits] = this_ph_data[0::WORDS_PER_HIT]
        hit_list[1, i : i + num_hits] = asic_num
        hit_list[2, i : i + num_hits] = channel_num
        hit_list[4, i : i + num_hits] = this_pkt_num
        hit_list[5, i : i + num_hits] = this_ph_data[0::WORDS_PER_HIT]
        time_s[i : i + num_hits] = pkt_s
        time_clk[i : i + num_hits] = pkt_clk
        if WORDS_PER_HIT == 4:
            hit_list[6, i : i + num_hits] = this_ph_data[2::WORDS_PER_HIT]  # baseline
            hit_list[3, i : i + num_hits] = this_ph_data[3::WORDS_PER_HIT]  # hit energy
        elif WORDS_PER_HIT == 3:
            hit_list[3, i : i + num_hits] = this_ph_data[2::WORDS_PER_HIT]  # hit energy
        i += num_hits

    event_list = Table()
    event_list["seqcount"] = hit_list[4, :]
    event_list["clocks"] = hit_list[5, :]
    event_list["asic"] = hit_list[1, :].astype(np.uint8)
    event_list["channel"] = hit_list[2, :].astype(np.uint8)
    event_list["atod"] = hit_list[3, :]
    event_list["baseline"] = hit_list[6, :]  # if baseline not present then all zeros
    event_list["pkttimes"] = time_s
    event_list["pktclock"] = time_clk
    date_beg = calc_time(time_s[0], time_clk[0])
    date_end = calc_time(time_s[-1], time_clk[-1])
    event_list.meta.update({"DATE-BEG": date_beg.fits})
    event_list.meta.update({"DATE-END": date_end.fits})
    center_index = int(len(time_s) / 2.0)
    date_avg = calc_time(time_s[center_index], time_clk[center_index])
    event_list.meta.update({"DATE-AVG": date_avg.fits})
    # event_list = TimeSeries(
    #    time=ph_times,
    #    data={
    #        "atod": hit_list[3, :],
    #        "asic": hit_list[1, :],
    #        "channel": hit_list[2, :],
    #        "clock": hit_list[5, :],
    #        "pktnum": hit_list[4, :],
    #        # "num": np.ones(len(hit_list[2, :])),
    #    }
    # )
    # event_list.sort()
    # int_time = (event_list.time.max() - event_list.time.min()).to("min")
    # event_list.meta.update({"int_time": int_time})
    # event_list.meta.update({"avg rate": (total_hits * u.ct) / int_time})
    return event_list, pkt_list


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
    packet_bytes = stream_by_apid.get(APID["housekeeping"], None)
    if packet_bytes is None:
        return None
    packet_definition = packet_definition_hk()
    pkt = ccsdspy.FixedLength(packet_definition)
    hk_data = pkt.load(packet_bytes, include_primary_header=True)
    hk_timestamps = [
        dt.timedelta(seconds=int(this_t)) + EPOCH for this_t in hk_data["timestamp"]
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
    packet_bytes = stream_by_apid.get(APID["spectrum"], None)
    if packet_bytes is None:
        return None
    packet_definition = packet_definition_hist2()
    pkt = ccsdspy.FixedLength(packet_definition)
    data = pkt.load(packet_bytes, include_primary_header=True)
    timestamps = calc_time(data["TIMESTAMPS"], data["TIMESTAMPCLOCK"])
    num_packets = len(timestamps)
    h = data["HISTOGRAM_DATA"].reshape((num_packets, 24, 513))
    histogram_data = h[:, :, 1:]  # remove the id field
    ids = h[:, :, 0]
    return timestamps, histogram_data, ids


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
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_bytes = stream_by_apid.get(APID["cmd_resp"], None)
    if packet_bytes is None:
        return None
    packet_definition = packet_definition_cmd_response()
    pkt = ccsdspy.FixedLength(packet_definition)
    data = pkt.load(packet_bytes, include_primary_header=True)
    timestamps = calc_time(data["TIMESTAMPS"], data["TIMESTAMPCLOCK"])
    data = {
        "time_s": data["TIMESTAMPS"],
        "time_clock": data["TIMESTAMPCLOCK"],
        "address": data["ADDR"],
        "value": data["VALUE"],
        "seqcount": data["CCSDS_SEQUENCE_COUNT"],
    }
    ts = TimeSeries(time=timestamps, data=data)
    return ts


def packet_definition_hk():
    """Return the packet definiton for the housekeeping packets."""
    hk_table = ascii.read(padre_meddea._data_directory / "hk_packet_def.csv")
    hk_table.add_index("name")
    p = [PacketField(name="timestamp", data_type="uint", bit_length=32)]
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


def packet_definition_cmd_response():
    """Return the packet definiton for the register read response"""
    p = [
        PacketField(name="TIMESTAMPS", data_type="uint", bit_length=32),
        PacketField(name="TIMESTAMPCLOCK", data_type="uint", bit_length=32),
        PacketField(name="ADDR", data_type="uint", bit_length=16),
        PacketField(name="VALUE", data_type="uint", bit_length=16),
        PacketField(name="CHECKSUM", data_type="uint", bit_length=16),
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
