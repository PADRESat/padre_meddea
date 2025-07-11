"""
This module provides a generic file reader.
"""

from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import ccsdspy
import numpy as np
from astropy.table import Table
from astropy.time import Time
from astropy.timeseries import TimeSeries
from ccsdspy import PacketArray, PacketField
from ccsdspy.utils import count_packets, split_by_apid
from specutils import Spectrum1D

import padre_meddea.util.util as util
from padre_meddea import APID, log
from padre_meddea.housekeeping.housekeeping import parse_housekeeping_packets
from padre_meddea.housekeeping.register import add_register_address_name
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
    # TODO: the following should use parse_science_filename
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
                case "padremdu8":  # housekeeping and command response data
                    result = read_raw_u8(this_path)
        case _:
            raise ValueError(f"File extension {this_path.suffix} not recognized.")
    return result


def read_raw_a2(filename: Path) -> SpectrumList:
    """
    Read a raw Spectrum (A2) packet file.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    spectra : SpectrumList
    """
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    pkt_spec, specs, pixel_ids = raw_data["spectra"]
    result = SpectrumList(pkt_spec, specs, pixel_ids)
    return result


def read_raw_a0(filename: Path) -> PhotonList:
    """
    Read a raw photon (A0) packet file.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    eventlist : PhotonList
    """
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    pkt_list, event_list = raw_data["photons"]
    result = PhotonList(pkt_list, event_list)
    return result


def read_raw_u8(filename: Path):
    """
    Read a raw housekeeping (U8) packet file.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    housekeeping timeseries, command timeseries
    """
    this_path = Path(filename)
    raw_data = read_raw_file(this_path)
    hk_ts = raw_data["housekeeping"]
    cmd_ts = raw_data["cmd_resp"]
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
    Read a fits file of any level and return the appropriate data objects.
    """
    hdul = fits.open(filename)
    header = hdul[0].header.copy()
    hdul.close()
    level = header["LEVEL"]
    data_type = header["BTYPE"]

    if level in ["l0", "l1"]:
        match data_type:
            case "photon":
                return read_fits_l0l1_photon(filename)
            case "housekeeping":
                return read_fits_l0l1_housekeeping(filename)
            case "spectrum":
                return read_fits_l0l1_spectrum(filename)
            case _:
                raise ValueError(f"Data type {data_type} is not recognized.")
    else:
        raise ValueError(
            f"File level, {level}, and data type, {data_type}, of {filename} not recogized."
        )


def read_fits_l0l1_photon(filename: Path) -> PhotonList:
    """
    Read a level 0 photon fits file.
    """
    event_list_table = Table.read(filename, hdu=1)
    ph_times = util.calc_time(
        event_list_table["pkttimes"],
        event_list_table["pktclock"],
        event_list_table["clocks"],
    )
    event_list_table["time"] = ph_times
    event_list = TimeSeries(event_list_table)

    packet_list_table = Table.read(filename, hdu=2)
    pkt_times = util.calc_time(
        packet_list_table["pkttimes"], packet_list_table["pktclock"]
    )
    packet_list_table["time"] = pkt_times
    packet_list = TimeSeries(packet_list_table)

    return PhotonList(packet_list, event_list)


def read_fits_l0l1_housekeeping(filename: Path) -> tuple[TimeSeries, TimeSeries]:
    """Read a level 0 housekeeping file

    Returns
    -------
    hk_ts, cmd_ts
        TimeSeries of housekeeping data, TimeSeries of reads
    """
    hk_table = Table.read(filename, hdu=1)
    hk_times = util.calc_time(hk_table["timestamp"])
    hk_table["time"] = hk_times
    hk_ts = TimeSeries(hk_table)

    cmd_table = Table.read(filename, hdu=2)
    if len(cmd_table) == 0:
        log.warning(f"No command response data found in {filename}")
        cmd_ts = TimeSeries()
    else:
        cmd_times = util.calc_time(cmd_table["pkttimes"], cmd_table["pktclock"])
        cmd_table["time"] = cmd_times
        cmd_ts = TimeSeries(cmd_table)

    return hk_ts, cmd_ts


def read_fits_l0l1_spectrum(filename: Path):
    """Read a level 0 spectrum file.

    .. note::
       This function is in Draft form and what it returns will likely be updated.

    Returns
    -------
    timestamps, Spectrum1D array, asic_nums, pixel_nums, pixelid_strings
    """
    pkt_table = Table.read(filename, hdu=2)
    pkt_times = util.calc_time(pkt_table["pkttimes"], pkt_table["pktclock"])
    pkt_table["time"] = pkt_times
    pkt_ts = TimeSeries(pkt_table)

    hdu = fits.open(filename)
    specs = Spectrum1D(
        spectral_axis=np.arange(512) * u.pix, flux=hdu["spec"].data * u.ct
    )
    # reconstruct pixel ids TODO use util.get_pixelid
    pixel_ids = (hdu["PKT"].data["asic"] << 5) + (hdu["PKT"].data["channel"]) + 0xCA00
    return SpectrumList(pkt_ts, specs, pixel_ids)


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
        "pkttimes": data["TIME_S"],
        "pktclock": data["TIME_CLOCKS"],
        "address": data["ADDR"],
        "value": data["VALUE"],
        "seqcount": data["CCSDS_SEQUENCE_COUNT"],
    }
    ts = TimeSeries(time=timestamps, data=data)
    ts = add_register_address_name(ts)
    ts.meta.update({"ORIGFILE": f"{filename.name}"})

    # Clean Command Response Times
    ts = clean_cmd_response_data(ts)

    return ts


def clean_cmd_response_data(ts: TimeSeries) -> TimeSeries:
    """Given raw command response packet data, perform a cleaning operation that removes bad data.
    The most common cause of which are bits that are turned to zero when they should not be.

    This function finds unphysical times (before 2024-01-01) and replaces the time with an estimated time
    by using the median time between command responses.

    Parameters
    ----------
    ts : TimeSeries
        A TimeSeries object containing the command response data.

    Returns
    -------
    TimeSeries
        A TimeSeries containing the cleaned command response data.
    """
    # Calculate Differences in Time-Related Columns
    dts = ts.time[1:] - ts.time[:-1]
    pkttimes_diff = ts["pkttimes"][1:] - ts["pkttimes"][:-1]
    pktclock_diff = ts["pktclock"][1:] - ts["pktclock"][:-1]

    # Calculate the Cadence to use for Interpolation
    median_dt = np.median(dts)
    pkttimes_diff_median = np.median(pkttimes_diff)
    pktclock_diff_median = np.median(pktclock_diff)

    bad_indices = np.argwhere(ts.time <= Time("2024-01-01T00:00"))
    for this_bad_index in bad_indices:
        if this_bad_index < len(ts.time) - 1:
            ts.time[this_bad_index] = ts.time[this_bad_index + 1] - median_dt
            ts["pkttimes"][this_bad_index] = (
                ts["pkttimes"][this_bad_index + 1] - pkttimes_diff_median
            )
            ts["pktclock"][this_bad_index] = (
                ts["pktclock"][this_bad_index + 1] - pktclock_diff_median
            )
        else:
            ts.time[this_bad_index] = ts.time[this_bad_index - 1] + median_dt
            ts["pkttimes"][this_bad_index] = (
                ts["pkttimes"][this_bad_index - 1] + pkttimes_diff_median
            )
            ts["pktclock"][this_bad_index] = (
                ts["pktclock"][this_bad_index - 1] + pktclock_diff_median
            )

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
