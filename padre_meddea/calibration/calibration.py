"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

from astropy.io import fits, ascii
from astropy.time import Time
from astropy.table import Table

from swxsoc.util.util import record_timeseries

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools, fits_tools

from padre_meddea.util.util import create_science_filename, calc_time
from padre_meddea.io.file_tools import read_raw_file

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

    if file_path.suffix == ".bin":
        parsed_data = read_raw_file(file_path)
        if parsed_data["photons"] is not None:  # we have event list data
            event_list, pkt_list = parsed_data["photons"]

            primary_hdr = fits.Header()

            # fill in metadata
            primary_hdr["DATE"] = (Time.now().fits, "FITS file creation date in UTC")
            for this_keyword, this_str in zip(
                ["DATE-BEG", "DATE-END", "DATE-AVG"],
                [
                    "Acquisition start time",
                    "Acquisition end time",
                    "Average time of acquisition",
                ],
            ):
                primary_hdr[this_keyword] = (
                    event_list.meta.get(this_keyword, ""),
                    this_str,
                )

            primary_hdr["LEVEL"] = (0, "Data level of fits file")

            # add processing information
            primary_hdr = add_process_info_to_header(primary_hdr)
            
            # custom keywords
            primary_hdr["DATATYPE"] = ("event_list", "Description of the data")
            primary_hdr["ORIGAPID"] = (padre_meddea.APID["photon"], "APID(s) of the originating data")
            primary_hdr["ORIGFILE"] = (file_path.name, "Originating file(s)")

            empty_primary = fits.PrimaryHDU(header=primary_hdr)
            pkt_hdu = fits.BinTableHDU(pkt_list, name="PKT")
            pkt_hdu.add_checksum()
            hit_hdu = fits.BinTableHDU(event_list, name="SCI")
            hit_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary, hit_hdu, pkt_hdu])

            path = create_science_filename(
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
            record_timeseries(hk_data, "housekeeping")
            hk_table = Table(hk_data)
            primary_hdr = fits.Header()
            # fill in metadata
            primary_hdr["DATE"] = (Time.now().fits, "FITS file creation date in UTC")
            primary_hdr["LEVEL"] = (0, "Data level of fits file")
            primary_hdr["DATATYPE"] = ("housekeeping", "Description of the data")
            primary_hdr["ORIGAPID"] = (padre_meddea.APID["housekeeping"], "APID(s) of the originating data")
            primary_hdr["ORIGFILE"] = (file_path.name, "Originating file(s)")
            date_beg = calc_time(hk_data['timestamp'][0])
            primary_hdr["DATEREF"] = (date_beg.fits, "Reference date")

            # add processing information
            primary_hdr = add_process_info_to_header(primary_hdr)

            # add common fits keywords
            fits_meta = read_fits_keyword_file(
                padre_meddea._data_directory / "fits_keywords_primaryhdu.csv"
            )
            for row in fits_meta:
                primary_hdr[row["keyword"]] = (row["value"], row["comment"])
            hk_table['seqcount'] = hk_table["CCSDS_SEQUENCE_COUNT"]
            colnames_to_remove = ["CCSDS_VERSION_NUMBER", "CCSDS_PACKET_TYPE", "CCSDS_SECONDARY_FLAG", "CCSDS_SEQUENCE_FLAG", "CCSDS_APID", "CCSDS_SEQUENCE_COUNT", "CCSDS_PACKET_LENGTH", "CHECKSUM", "time"]
            for this_col in colnames_to_remove:
                if this_col in hk_table.colnames:
                    hk_table.remove_column(this_col)

            empty_primary = fits.PrimaryHDU(header=primary_hdr)
            hk_hdu = fits.BinTableHDU(hk_table, name="HK")
            hk_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary, hk_hdu])

            path = create_science_filename(
                time=date_beg,
                level="l1",
                descriptor="hk",
                test=True,
                version="0.1.0",
            )
            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)

            


    #  calibrated_file = calibrate_file(data_filename)
    #  data_plot_files = plot_file(data_filename)
    #  calib_plot_files = plot_file(calibrated_file)

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
