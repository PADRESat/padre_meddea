"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

import numpy as np

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.timeseries import TimeSeries

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools, fits_tools
from padre_meddea.util import util, validation
import padre_meddea.io.aws_db as aws_db

from padre_meddea.util.util import create_science_filename, calc_time
from padre_meddea.io.file_tools import read_raw_file
from padre_meddea.io.fits_tools import (
    add_process_info_to_header,
    get_primary_header,
    get_std_comment,
)

__all__ = [
    "process_file",
    "get_calibration_file",
    "read_calibration_file",
]


def process_file(filename: Path, overwrite=False) -> list:
    """
    This is the entry point for the pipeline processing.
    It runs all of the various processing steps required.

    Parameters
    ----------
    data_filename: str
        Fully specificied filename of an input file

    Returns
    -------
    output_filenames: list
        Fully specificied filenames for the output files.
    """
    log.info(f"Processing file {filename}.")
    # Check if the LAMBDA_ENVIRONMENT environment variable is set
    lambda_environment = os.getenv("LAMBDA_ENVIRONMENT")
    output_files = []
    file_path = Path(filename)

    if file_path.suffix.lower() in [".bin", ".dat"]:
        # Before we process, validate the file with CCSDS
        custom_validators = [validation.validate_packet_checksums]
        validation_findings = validation.validate(
            file_path,
            valid_apids=list(padre_meddea.APID.values()),
            custom_validators=custom_validators,
        )
        for finding in validation_findings:
            log.warning(f"Validation Finding for File : {filename} : {finding}")

        parsed_data = read_raw_file(file_path)
        if parsed_data["photons"] is not None:  # we have event list data
            pkt_list, event_list = parsed_data["photons"]
            log.info(
                f"Found photon data, {len(event_list)} photons and {len(pkt_list)} packets."
            )
            aws_db.record_photons(pkt_list, event_list)

            event_list = Table(event_list)
            event_list.remove_column("time")
            primary_hdr = get_primary_header()
            primary_hdr = add_process_info_to_header(primary_hdr)
            primary_hdr["LEVEL"] = (0, get_std_comment("LEVEL"))
            primary_hdr["DATATYPE"] = ("event_list", get_std_comment("DATATYPE"))
            primary_hdr["ORIGAPID"] = (
                padre_meddea.APID["photon"],
                get_std_comment("ORIGAPID"),
            )
            primary_hdr["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))

            for this_keyword in ["DATE-BEG", "DATE-END", "DATE-AVG"]:
                primary_hdr[this_keyword] = (
                    event_list.meta.get(this_keyword, ""),
                    get_std_comment(this_keyword),
                )

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            pkt_list = Table(pkt_list)
            pkt_list.remove_column("time")
            pkt_hdu = fits.BinTableHDU(pkt_list, name="PKT")
            pkt_hdu.add_checksum()
            hit_hdu = fits.BinTableHDU(event_list, name="SCI")
            hit_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary_hdu, hit_hdu, pkt_hdu])

            path = create_science_filename(
                "meddea",
                time=primary_hdr["DATE-BEG"],
                level="l1",
                descriptor="eventlist",
                test=True,
                version="0.1.0",
            )

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            # Write the file, with the overwrite option controlled by the environment variable
            hdul.writeto(path, overwrite=overwrite)
            # Store the output file path in a list
            output_files.append(path)
        if parsed_data["housekeeping"] is not None:
            hk_data = parsed_data["housekeeping"]
            # send data to AWS Timestream for Grafana dashboard
            aws_db.record_housekeeping(hk_data)
            hk_table = Table(hk_data)

            primary_hdr = get_primary_header()
            primary_hdr = add_process_info_to_header(primary_hdr)
            primary_hdr["LEVEL"] = (0, get_std_comment("LEVEL"))
            primary_hdr["DATATYPE"] = ("housekeeping", get_std_comment("DATATYPE"))
            primary_hdr["ORIGAPID"] = (
                padre_meddea.APID["housekeeping"],
                get_std_comment("ORIGAPID"),
            )
            primary_hdr["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))

            date_beg = calc_time(hk_data["timestamp"][0])
            primary_hdr["DATEREF"] = (date_beg.fits, get_std_comment("DATEREF"))

            hk_table["seqcount"] = hk_table["CCSDS_SEQUENCE_COUNT"]
            colnames_to_remove = [
                "CCSDS_VERSION_NUMBER",
                "CCSDS_PACKET_TYPE",
                "CCSDS_SECONDARY_FLAG",
                "CCSDS_SEQUENCE_FLAG",
                "CCSDS_APID",
                "CCSDS_SEQUENCE_COUNT",
                "CCSDS_PACKET_LENGTH",
                "CHECKSUM",
                "time",
            ]
            for this_col in colnames_to_remove:
                if this_col in hk_table.colnames:
                    hk_table.remove_column(this_col)

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hk_hdu = fits.BinTableHDU(data=hk_table, name="HK")
            hk_hdu.add_checksum()

            # add command response data if it exists  in the same fits file
            if parsed_data["cmd_resp"] is not None:
                data_ts = parsed_data["cmd_resp"]
                this_header = fits.Header()
                this_header["DATEREF"] = (
                    data_ts.time[0].fits,
                    get_std_comment("DATEREF"),
                )
                aws_db.record_cmd(data_ts)
                data_table = Table(data_ts)
                colnames_to_remove = [
                    "CCSDS_VERSION_NUMBER",
                    "CCSDS_PACKET_TYPE",
                    "CCSDS_SECONDARY_FLAG",
                    "CCSDS_SEQUENCE_FLAG",
                    "CCSDS_APID",
                    "CCSDS_SEQUENCE_COUNT",
                    "CCSDS_PACKET_LENGTH",
                    "CHECKSUM",
                    "time",
                ]
                for this_col in colnames_to_remove:
                    if this_col in hk_table.colnames:
                        data_table.remove_column(this_col)
                cmd_hdu = fits.BinTableHDU(data=data_table, name="READ")
                cmd_hdu.add_checksum()
            else:  # if None still end an empty Binary Table
                this_header = fits.Header()
                cmd_hdu = fits.BinTableHDU(data=None, header=this_header, name="READ")
            hdul = fits.HDUList([empty_primary_hdu, hk_hdu, cmd_hdu])

            path = create_science_filename(
                "meddea",
                time=date_beg,
                level="l1",
                descriptor="hk",
                test=True,
                version="0.1.0",
            )

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)
        if parsed_data["spectra"] is not None:
            ts, spectra, ids = parsed_data["spectra"]
            aws_db.record_spectra(ts, spectra, ids)
            asic_nums, channel_nums = util.parse_pixelids(ids)
            # asic_nums = (ids & 0b11100000) >> 5
            # channel_nums = ids & 0b00011111
            # TODO check that asic_nums and channel_nums do not change

            primary_hdr = get_primary_header()
            primary_hdr = add_process_info_to_header(primary_hdr)
            primary_hdr["LEVEL"] = (0, get_std_comment("LEVEL"))
            primary_hdr["DATATYPE"] = ("spectrum", get_std_comment("DATATYPE"))
            primary_hdr["ORIGAPID"] = (
                padre_meddea.APID["spectrum"],
                get_std_comment("ORIGAPID"),
            )
            primary_hdr["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))
            dates = {
                "DATE-BEG": ts.time[0].fits,
                "DATE-END": ts.time[-1].fits,
                "DATE-AVG": ts.time[len(ts.time) // 2].fits,
            }
            primary_hdr["DATEREF"] = (dates["DATE-BEG"], get_std_comment("DATEREF"))
            for this_keyword, value in dates.items():
                primary_hdr[this_keyword] = (
                    value,
                    get_std_comment(this_keyword),
                )
            spec_hdu = fits.ImageHDU(data=spectra.data, name="SPEC")
            spec_hdu.add_checksum()

            data_table = Table()
            data_table["pkttimes"] = ts["pkttimes"]
            data_table["pktclock"] = ts["pktclock"]
            data_table["asic"] = asic_nums
            data_table["channel"] = channel_nums
            data_table["seqcount"] = ts["seqcount"]

            pkt_hdu = fits.BinTableHDU(data=data_table, name="PKT")
            pkt_hdu.add_checksum()
            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hdul = fits.HDUList([empty_primary_hdu, spec_hdu, pkt_hdu])
            path = create_science_filename(
                "meddea",
                time=dates["DATE-BEG"],
                level="l1",
                descriptor="spec",
                test=True,
                version="0.1.0",
            )

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)

    # add other tasks below
    return output_files


def raw_to_l0(filename: Path):
    if not (filename.suffix == "bin"):
        raise ValueError(f"File {filename} extension not recognized.")

    data = file_tools.read_raw_file(filename)


def get_calibration_file(time: Time) -> Path:
    """
    Given a time, return the appropriate calibration file.

    Parameters
    ----------
    data_filename: str
        Fully specificied filename of the non-calibrated file (data level < 2)
    time: ~astropy.time.Time

    Returns
    -------
    calib_filename: str
        Fully specificied filename for the appropriate calibration file.

    Examples
    --------
    """
    return None


def read_calibration_file(calib_filename: Path):
    """
    Given a calibration, return the calibration structure.

    Parameters
    ----------
    calib_filename: str
        Fully specificied filename of the non-calibrated file (data level < 2)

    Returns
    -------
    output_filename: str
        Fully specificied filename of the appropriate calibration file.

    Examples
    --------
    """

    # if can't read the file

    return None
