"""
This module provides general utility functions.
"""

import os

from astropy.time import Time

import padre_meddea


__all__ = ["create_science_filename", "parse_science_filename", "VALID_DATA_LEVELS"]

TIME_FORMAT_L0 = "%Y%j-%H%M%S"
TIME_FORMAT = "%Y%m%dT%H%M%S"
VALID_DESCRIPTORS = ["eventlist", "spec-eventlist", "spec", "xraydirect"]
VALID_DATA_LEVELS = ["l0", "l1", "l2", "l3", "l4"]
FILENAME_EXTENSION = ".fits"


def create_science_filename(
    instrument: str,
    time: str,
    level: str,
    version: str,
    descriptor: str,
    mode: str = "",
    test: bool = False,
):
    """Return a compliant filename. The format is defined as

    padre_{instrument}_{mode}_{level}{test}_{descriptor}_{time}_v{version}.cdf

    This format is only appropriate for data level >= 1.

    Parameters
    ----------
    instrument : `str`
        The instrument name.
    time : `str` (in isot format) or ~astropy.time
        The time
    level : `str`
        The data level. Must be one of the following "l0", "l1", "l2", "l3", "l4", "ql"
    version : `str`
        The file version which must be given as X.Y.Z
    descriptor : `str`
        An optional file descriptor.
    mode : `str`
        An optional instrument mode.
    test : bool
        Selects whether the file is a test file.

    Returns
    -------
    filename : `str`
        A file name including the given parameters that matches the PADRE file naming conventions

    Raises
    ------
    ValueError: If the instrument is not recognized as one of the PADRE instruments
    ValueError: If the data level is not recognized as one of the PADRE valid data levels
    ValueError: If the data version does not match the PADRE data version formatting conventions
    ValueError: If the data product descriptor or instrument mode do not match the PADRE formatting conventions
    """
    test_str = ""
    if instrument not in ["meddea"]:
        raise ValueError(f"Instrument, {instrument}, is not recognized.")

    if isinstance(time, str):
        time_str = Time(time, format="isot").strftime(TIME_FORMAT)
    else:
        time_str = time.strftime(TIME_FORMAT)

    if level not in VALID_DATA_LEVELS:
        raise ValueError(
            f"Level, {level}, is not recognized. Must be one of {VALID_DATA_LEVELS[1:]}."
        )
    # check that version is in the right format with three parts
    if len(version.split(".")) != 3:
        raise ValueError(
            f"Version, {version}, is not formatted correctly. Should be X.Y.Z"
        )
    # check that version has integers in each part
    for item in version.split("."):
        try:
            int_value = int(item)
        except ValueError:
            raise ValueError(f"Version, {version}, is not all integers.")

    if test is True:
        test_str = "test"

    # check that the descriptor is valid
    if descriptor not in VALID_DESCRIPTORS:
        raise ValueError(f"Descriptor, {descriptor}, is not recognized.")

    # the parse_science_filename function depends on _ not being present elsewhere
    if ("_" in mode) or ("_" in descriptor):
        raise ValueError(
            "The underscore symbol _ is not allowed in mode or descriptor."
        )

    filename = (
        f"padre_meddea_{mode}_{level}{test_str}_{descriptor}_{time_str}_v{version}"
    )
    filename = filename.replace("__", "_")  # reformat if mode or descriptor not given

    return filename + FILENAME_EXTENSION


def parse_science_filename(filepath: str) -> dict:
    """
    Parses a science filename into its consitutient properties (instrument, mode, test, time, level, version, descriptor).

    Parameters
    ----------
    filepath: `str`
        Fully specificied filepath of an input file

    Returns
    -------
    result : `dict`
        A dictionary with each property.

    Raises
    ------
    ValueError: If the file's mission name is not "PADRE"
    ValueError: If the file's instreument name is not one of the HERMES instruments
    ValueError: If the data level >0 for packet files
    ValueError: If not a CDF File
    """

    result = {
        "instrument": None,
        "mode": None,
        "test": False,
        "time": None,
        "level": None,
        "version": None,
        "descriptor": None,
    }

    filename = os.path.basename(filepath)
    file_name, file_ext = os.path.splitext(filename)

    filename_components = file_name.split("_")

    if filename_components[0] != "padre":
        raise ValueError(f"File {filename} not recognized. Not a valid mission name.")

    result["instrument"] = filename_components[1]

    if file_ext == ".bin":
        return {}  # TODO finish this
    elif file_ext == ".fits":
        if filename_components[1] not in "meddea":
            raise ValueError(
                "File {filename} not recognized. Not a valid instrument name."
            )

        result["time"] = Time.strptime(filename_components[-2], TIME_FORMAT)

        # mode and descriptor are optional so need to figure out if one or both or none is included
        if filename_components[2][0:2] not in VALID_DATA_LEVELS:
            # if the first component is not data level then it is mode and the following is data level
            result["mode"] = filename_components[2]
            result["level"] = filename_components[3].replace("test", "")
            if "test" in filename_components[3]:
                result["test"] = True
            if len(filename_components) == 7:
                result["descriptor"] = filename_components[4]
        else:
            result["level"] = filename_components[2].replace("test", "")
            if "test" in filename_components[2]:
                result["test"] = True
            if len(filename_components) == 6:
                result["descriptor"] = filename_components[3]
    else:
        raise ValueError(f"File extension {file_ext} not recognized.")

    result["instrument"] = filename_components[1]
    result["version"] = filename_components[-1][1:]  # remove the v

    return result
