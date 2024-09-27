"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

from astropy.io import fits, ascii
from astropy.time import Time
from astropy.table import Table

from swxsoc.util.util import create_science_filename

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools

# from padre_meddea.util.util import create_science_filename
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

    output_files = []

    if filename.suffix == ".bin":
        parsed_data = read_raw_file(filename)
        if "photons" in parsed_data.keys():  # we have event list data
            ph_list = parsed_data["photons"]
            hdu = fits.PrimaryHDU(data=None)
            hdu.header["DATE"] = (Time.now().fits, "FITS file creation date in UTC")
            fits_meta = read_fits_keyword_file(
                padre_meddea._data_directory / "fits_keywords_primaryhdu.csv"
            )
            for row in fits_meta:
                hdu.header[row["keyword"]] = (row["value"], row["comment"])
            bin_hdu = fits.BinTableHDU(data=Table(ph_list))
            hdul = fits.HDUList([hdu, bin_hdu])

            path = create_science_filename(
                "meddea",
                ph_list["time"][0].fits,
                "l1",
                descriptor="eventlist",
                test=True,
                version="0.1.0",
            )

            # Check if the LAMBDA_ENVIRONMENT environment variable is set
            lambda_environment = os.getenv("LAMBDA_ENVIRONMENT")

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            # Write the file, with the overwrite option controlled by the environment variable
            hdul.writeto(path, overwrite=overwrite)

            # Store the output file path in a list
            output_files = [path]

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


def read_fits_keyword_file(csv_file: Path):
    """Read csv file with default fits metadata information."""
    fits_meta_table = ascii.read(
        padre_meddea._data_directory / "fits_keywords_primaryhdu.csv",
        format="csv",
    )
    return fits_meta_table
