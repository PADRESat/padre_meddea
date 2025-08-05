"""
Provides generic file readers.
"""

from pathlib import Path

import astropy.io.fits as fits
import astropy.units as u
import numpy as np
from astropy.table import Table
from astropy.timeseries import TimeSeries
from ccsdspy.utils import count_packets, split_by_apid
from specutils import Spectrum1D

import padre_meddea.util.util as util
from padre_meddea import APID, log
from padre_meddea.housekeeping.housekeeping import (
    clean_hk_data,
    parse_cmd_response_packets,
    parse_housekeeping_packets,
)
from padre_meddea.spectrum.raw import (
    clean_spectra_data,
    parse_ph_packets,
    parse_spectrum_packets,
)
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
    packet_list = util.trim_timeseries(packet_list)
    event_list = util.trim_timeseries(event_list)

    return PhotonList(packet_list, event_list)


def read_fits_l0l1_housekeeping(filename: Path) -> tuple[TimeSeries, TimeSeries]:
    """Read a level 0 housekeeping file

    Returns
    -------
    hk_ts, cmd_ts
        TimeSeries of housekeeping data, TimeSeries of reads
    """
    hk_table = Table.read(filename, hdu=1)
    if "pkttimes" in hk_table.columns:
        hk_times = util.calc_time(hk_table["pkttimes"])
    elif "timestamp" in hk_table.columns:
        hk_times = util.calc_time(hk_table["timestamp"])
    hk_table["time"] = hk_times
    hk_ts = TimeSeries(hk_table)
    hk_ts = clean_hk_data(hk_ts)
    cmd_table = Table.read(filename, hdu=2)
    if len(cmd_table) == 0:
        log.warning(f"No command response data found in {filename}")
        cmd_ts = TimeSeries()
    else:
        cmd_times = util.calc_time(cmd_table["pkttimes"], cmd_table["pktclock"])
        cmd_table["time"] = cmd_times
        cmd_ts = TimeSeries(cmd_table)
        cmd_ts = util.trim_timeseries(cmd_ts)

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
    pkt_ts, specs, pixel_ids = clean_spectra_data(pkt_ts, specs, pixel_ids)

    return SpectrumList(pkt_ts, specs, pixel_ids)


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
