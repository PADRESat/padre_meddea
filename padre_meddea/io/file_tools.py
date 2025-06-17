"""
This module provides a generic file reader.
"""

from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.timeseries import TimeSeries
from astropy.time import Time

import ccsdspy
from ccsdspy import PacketField, PacketArray
from ccsdspy.utils import (
    count_packets,
    split_by_apid,
)
import astropy.io.fits as fits

from specutils import Spectrum1D

from padre_meddea import log
from padre_meddea import APID
import padre_meddea.util.util as util
from padre_meddea.housekeeping.housekeeping import parse_housekeeping_packets
from padre_meddea.spectrum.raw import parse_ph_packets, parse_spectrum_packets
from padre_meddea.spectrum.spectrum import PhotonList, SpectrumList

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
    this_path = Path(filename)
    match this_path.suffix.lower():
        case ".bin":  # raw binary file
            result = read_raw_file(this_path)
        case ".fits":  # level 0 or above
            result = read_fits(this_path)
        case ".dat":
            match this_path.name.lower()[0:9].lower():
                case "padremda2":  # summary spectrum data
                    result = read_raw_a2(this_path)
                case "padremda0":  # photon data
                    result = read_raw_a0(this_path)
                case "padremdu8": # housekeeping and command response data
                    result = read_raw_u8(this_path)
        case _:
            raise ValueError(f"File extension {this_path.suffix} not recognized.")
    return result


def read_raw_a2(filename: Path) -> SpectrumList:
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    pkt_spec, specs, pixel_ids = raw_data['spectra']
    result = SpectrumList(pkt_spec, specs, pixel_ids)
    return result


def read_raw_a0(filename: Path) -> PhotonList:
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    pkt_list, event_list = raw_data['photons']
    result = PhotonList(pkt_list, event_list)
    return result


def read_raw_u8(filename: Path):
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    hk_ts = raw_data['housekeeping']
    cmd_ts = raw_data['cmd_resp']
    return hk_ts, cmd_ts


def read_raw_file(filename: Path):
    """
    Read a raw data file.

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
        "housekeeping": parse_housekeeping_packets(filename),
        "spectra": parse_spectrum_packets(filename),
        "cmd_resp": parse_cmd_response_packets(filename),
    }

    return result


def read_fits(filename: Path):
    """
    Read a fits file.
    """
    hdul = fits.open(filename)
    header = hdul[0].header.copy()
    hdul.close()
    level = header["LEVEL"]
    data_type = header["DATATYPE"]

    if (level == 0) and (data_type == "event_list"):
        return read_fits_l0_event_list(filename)
    if (level == 0) and (data_type == "housekeeping"):
        return read_fits_l0_housekeeping(filename)
    if (level == 0) and (data_type == "spectrum"):
        return read_fits_l0_spectrum(filename)
    else:
        raise ValueError(f"File contents of {filename} not recogized.")


def read_fits_l0_event_list(filename: Path) -> TimeSeries:
    """ """
    with fits.open(filename) as hdu:
        # parse event data in SCI
        num_events = len(hdu["SCI"].data["seqcount"])
        ph_times = util.calc_time(
            hdu["sci"].data["pkttimes"],
            hdu["sci"].data["pktclock"],
            hdu["sci"].data["clocks"],
        )
        # add the pixel conversions
        pixels = util.channel_to_pixel(hdu["sci"].data["channel"])
        pixel_strs = [
            util.get_pixel_str(this_asic, this_pixel)
            for this_asic, this_pixel in zip(hdu["sci"].data["asic"], pixels)
        ]
        event_list = TimeSeries(
            time=ph_times,
            data={
                "atod": hdu["sci"].data["atod"],
                "baseline": hdu["sci"].data["baseline"],
                "asic": hdu["sci"].data["asic"],
                "channel": hdu["sci"].data["channel"],
                "pixel": pixels,
                "clocks": hdu["sci"].data["clocks"],
                "seqcount": hdu["sci"].data["seqcount"],
                "pixel_str": pixel_strs,
            },
        )
        event_list.sort()
        # parse packet header data
        pkt_times = util.calc_time(
            hdu["pkt"].data["pkttimes"], hdu["pkt"].data["pktclock"]
        )
        pkt_ts = TimeSeries(
            time=pkt_times,
            data={
                "livetime": hdu["pkt"].data["livetime"],
                "inttime": hdu["pkt"].data["inttime"],
                "flags": hdu["pkt"].data[
                    "flags"
                ],  # TODO: parse flags into individual columns
                "seqcount": hdu["pkt"].data["seqcount"],
            },
        )
    return event_list, pkt_ts


def read_fits_l0_housekeeping(filename: Path) -> TimeSeries:
    """Read a level 0 housekeeping file

    Returns
    -------
    TimeSeries of housekeeping data.
    """
    with fits.open(filename) as hdu:
        colnames = [this_col.name for this_col in hdu["HK"].data.columns]
        times = util.calc_time(hdu["HK"].data["timestamp"])
        hk_list = TimeSeries(
            time=times, data={key: hdu["hk"].data[key] for key in colnames}
        )
        return hk_list


def read_fits_l0_spectrum(filename: Path):
    """Read a level 0 spectrum file.

    .. note::
       This function is in Draft form and what it returns will likely be updated.

    Returns
    -------
    timestamps, Spectrum1D array, asic_nums, pixel_nums, pixelid_strings
    """
    with fits.open(filename) as hdu:
        timestamps = util.calc_time(
            hdu["PKT"].data["pkttimes"], hdu["PKT"].data["pktclock"]
        )
        asic_nums = hdu["PKT"].data["asic"]
        channel_nums = hdu["PKT"].data["channel"]
        pixel_nums = util.channel_to_pixel(channel_nums)
        these_asics = asic_nums[0]
        these_pixels = pixel_nums[0]
        pixel_strs = [
            util.get_pixel_str(this_asic, this_pixel)
            for this_asic, this_pixel in zip(these_asics, these_pixels)
        ]
        # TODO: check that all asic_nums and channel_nums are the same
        specs = Spectrum1D(
            spectral_axis=np.arange(512) * u.pix, flux=hdu["spec"].data * u.ct
        )
        ts = TimeSeries(times=timestamps)
        ts["asic"] = these_asics
        ts["pixel"] = these_pixels
        ts["pixel_str"] = pixel_strs
        ts["seqcount"] = hdu["pkt"].data["seqcount"]
        ts["pkttimes"] = hdu["pkt"].data["pkttimes"]

    return ts, specs


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
    timestamps = util.calc_time(data["TIME_S"], data["TIME_CLOCKS"])
    data = {
        "time_s": data["TIME_S"],
        "time_clock": data["TIME_CLOCKS"],
        "address": data["ADDR"],
        "value": data["VALUE"],
        "seqcount": data["CCSDS_SEQUENCE_COUNT"],
    }
    ts = TimeSeries(time=timestamps, data=data)
    ts.meta.update({"ORIGFILE": f"{filename.name}"})
    return ts


def packet_definition_cmd_response():
    """Return the packet definiton for the register read response"""
    p = [
        PacketField(name="TIME_S", data_type="uint", bit_length=32),
        PacketField(name="TIME_CLOCKS", data_type="uint", bit_length=32),
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


def clean_spectra_data(parsed_data):
    """Given raw spectrum packet data, perform a cleaning operation that removes bad data.
    The most common cause of which are bits that are turned to zero when they should not be.

    This function finds unphysical times (before 2022-01-01) and replaces the time with an estimated time by using the median time between spectra.
    It also replaces all pixel ids with the median pixel id set.

    Returns
    -------
    cleaned_parsed_data
    """
    ts, spectra, ids = parsed_data
    # remove bad times
    dts = ts.time[1:] - ts.time[:-1]
    median_dt = np.median(dts)
    bad_indices = np.argwhere(ts.time <= Time("2024-01-01T00:00"))
    for this_bad_index in bad_indices:
        if this_bad_index < len(ts.time) - 1:
            ts.time[this_bad_index] = ts.time[this_bad_index + 1] - median_dt
        else:
            ts.time[this_bad_index] = ts.time[this_bad_index - 1] + median_dt

    # remove bad pixel ids
    median_ids = np.median(ids, axis=0)
    fixed_ids = (
        np.tile(median_ids, ids.shape[0])
        .reshape(ids.shape[0], len(median_ids))
        .astype(ids.dtype)
    )
    return ts, spectra, fixed_ids
