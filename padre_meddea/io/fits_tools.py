"""
This module provides a utilities to manage fits files reading and writing.
"""

import gc
import json
import os
import re
import tempfile
import time
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Tuple

import astropy.io.fits as fits
import ccsdspy
import git
import numpy as np
import solarnet_metadata
from astropy import units as u
from collections import defaultdict
from datetime import datetime, timedelta, time

from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.utils.metadata import MergeConflictWarning
from solarnet_metadata.schema import SOLARNETSchema

import padre_meddea
from padre_meddea import log
from padre_meddea.util.util import (
    calc_time,
    create_science_filename,
    parse_science_filename,
)

CUSTOM_ATTRS_PATH = (
    padre_meddea._data_directory / "fits" / "fits_keywords_primaryhdu.yaml"
)

FITS_HDR_KEYTOCOMMENT = ascii.read(
    padre_meddea._data_directory / "fits" / "fits_keywords_dict.csv", format="csv"
)
FITS_HDR_KEYTOCOMMENT.add_index("keyword")

# Dict[str, Tuple[str, str, str]]
# Lib <n><a>: (Library Name, Library Version, Library URL)
PRLIBS = {
    "1A": (
        "padre_meddea",
        padre_meddea.__version__,
        "https://github.com/PADRESat/padre_meddea.git",
    ),
    "1B": ("ccsdspy", ccsdspy.__version__, "https://github.com/CCSDSPy/ccsdspy.git"),
    "1C": (
        "solarnet_metadata",
        solarnet_metadata.__version__,
        "https://github.com/IHDE-Alliance/solarnet_metadata.git",
    ),
}


def get_comment(keyword: str) -> str:
    """Given a keyword, return the standard comment for a header card."""
    # Check Keyword in Existing Structure
    if keyword.upper() in FITS_HDR_KEYTOCOMMENT["keyword"]:
        return FITS_HDR_KEYTOCOMMENT.loc[keyword]["comment"]
    # Check if it's an iterable keyword
    for this_row in FITS_HDR_KEYTOCOMMENT:
        res = re.fullmatch(this_row["keyword"], keyword)
        if res:
            comment = this_row["comment"]
            if len(res.groupdict()) > 0:  # check if there was a match
                for key, value in res.groupdict().items():
                    comment = comment.replace(f"<{key}>", value)
            return comment
    # No Match in Existing Data Structure - check SOLARNET Schema
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])
    if keyword in schema.attribute_schema["attribute_key"]:
        keyword_info = schema.attribute_schema["attribute_key"][keyword]
        comment = keyword_info["human_readable"]
        return comment
    return ""  # if nothing is found return empty string


def get_primary_header(
    file_path: Path, data_level: str, data_type: str, procesing_step: int = 1
) -> fits.Header:
    """
    Create a standard FITS primary header following SOLARNET conventions.

    This function creates a new FITS header with standard metadata including the
    current date, default PADRE attributes, processing information, data level,
    data type, original APID, and original filename.

    Parameters
    ----------
    file_path : Path
        Path to the original file, used to extract the filename
    data_level : str
        Data Processing step (e.g., 'L0', 'L1')
    data_type : str
        Type of data being processed
    procesing_step : int, default 1
        Processing step number to be added to header metadata

    Returns
    -------
    fits.Header
        A FITS header populated with standard metadata
    """
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])

    # Create a new header
    header = fits.Header()
    header["DATE"] = (Time.now().fits, get_comment("DATE"))

    # Add PADRE Default Attributes to Header
    for keyword, value in schema.default_attributes.items():
        header[keyword] = (value, get_comment(keyword))

    # FITS Standard Keywords
    header["EXTEND"] = ("T", get_comment("EXTEND"))

    # Data Description Keywords
    header["BTYPE"] = (data_type, get_comment("BTYPE"))

    # Pipeline processing keywords
    header = add_process_info_to_header(header, n=procesing_step)
    header["LEVEL"] = (data_level, get_comment("LEVEL"))

    # PADRE Custom Keywords
    header["ORIGAPID"] = (
        padre_meddea.APID[data_type],
        get_comment("ORIGAPID"),
    )
    file_path = Path(file_path)
    header["ORIGFILE"] = (file_path.name, get_comment("ORIGFILE"))

    return header


def get_obs_header(
    data_level: str,
    data_type: str,
):
    """
    Create a standard FITS header for the observation.

    This function creates a new FITS header with standard metadata including the
    current date, default PADRE attributes, processing information, data level,
    data type, original APID, and original filename.

    Parameters
    ----------
    data_level : str
        Data Processing step (e.g., 'L0', 'L1')
    data_type : str
        Type of data being processed

    Returns
    -------
    fits.Header
        A FITS header populated with standard metadata
    """
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])

    # Create a new header
    header = fits.Header()
    header["DATE"] = (Time.now().fits, get_comment("DATE"))

    # Add PADRE Default Attributes to Header
    for keyword, value in schema.default_attributes.items():
        header[keyword] = (value, get_comment(keyword))

    # Add PADRE Default Attributes to Header
    header["OBS_HDU"] = (1, get_comment("OBS_HDU"))
    header["SOLARNET"] = (
        0.5,
        get_comment("SOLARNET"),
    )  # I'll say we're partially compliant
    header["ORIGIN"]

    # Data Description Keywords
    header["BTYPE"] = (data_type, get_comment("BTYPE"))
    header["BUNIT"] = get_bunit(data_level, data_type)

    return header


# =============================================================================
# Mandatory data description keywords (sections 15.4, 5.1, 5.2, 5.6.2)
# =============================================================================


def get_bunit(data_level: str, data_type: str) -> Tuple[str, str]:
    """
    Get the bunit and comment for a given data level and data type.

    Parameters
    ----------
    data_level : str
        Data Processing step (e.g., 'L0', 'L1')
    data_type : str
        Type of data being processed

    Returns
    -------
    tuple
        A tuple of (bunit, comment)
    """
    bunit = None
    match data_level.lower():
        case "l0":
            bunit = u.dimensionless_unscaled
        case "l1":
            match data_type.lower():
                case "photon":
                    bunit = u.count
                case "housekeeping":
                    bunit = u.count
                case "spectrum":
                    bunit = u.count
                case _:
                    raise ValueError(f"Units Undefined for Data Type: {data_type}")
        case _:
            raise ValueError(f"Units Undefined for Data Level: {data_level}")
    comment = get_comment("BUNIT")
    return bunit.to_string(), comment


# =============================================================================
# Optional pipeline processing keywords (sections 18, 8, 8.1, 8.2)
# =============================================================================


def add_process_info_to_header(header: fits.Header, n: int = 1) -> fits.Header:
    """Add processing info metadata to a fits header.

    It adds the following SOLARNET compatible FITS cards;
    PRSTEPn, PRPROCn, PRPVERn, PRLIBnA, PRVERnA, PRLIBnA, PRHSHnA, PRVERnB

    Parameters
    ----------
    header : fits.Header
        The fits header to add the new cards to
    n : int, default 1
        The processing step number. Must be >= 1 and <= 9.

    Returns
    -------
    header : fits.Header
    """
    if n < 1 or n > 10:
        raise ValueError(f"Processing number, n, must be in range 1<=n<=9. Got {n}")

    # Pipeline-Level Metadata
    header[f"PRSTEP{n}"] = get_prstep(n)
    header[f"PRPROC{n}"] = get_prproc(n)
    header[f"PRPVER{n}"] = get_prpver(n)

    # Library-Level Metadata
    # Get Libraries used for the Given Level
    level_libraries = [key[1] for key in PRLIBS.keys() if key[0] == str(n)]
    for a in level_libraries:
        header[f"PRLIB{n}{a}"] = get_prlib(n, a)
        header[f"PRVER{n}{a}"] = get_prver(n, a)
        header[f"PRHSH{n}{a}"] = get_prhsh(n, a)

    #  primary_hdr["PRLOG1"] add log information, need to do this after the fact
    #  primary_hdr["PRENV1"] add information about processing env, need to do this after the fact
    return header


def get_prstep(n: int = 1) -> Tuple[str, str]:
    """
    Get the processing step description and standard comment for FITS header.
    This function returns a tuple containing the processing step description and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.
        1: Raw to L1
        2: L1 to L2
        3: L2 to L3
        4: L3 to L4

    Returns
    -------
    tuple
        A tuple of (processing_step_description, standard_comment)

    Raises
    ------
    ValueError
        If the Processing step number is not in the range 1-4.
    """
    value = None
    match n:
        case 1:
            value = "PROCESS Raw to L1"
        case 2:
            value = "PROCESS L1 to L2"
        case 3:
            value = "PROCESS L2 to L3"
        case 4:
            value = "PROCESS L3 to L4"
        case _:
            raise ValueError(f"Processing Undefined for n={n}")
    comment = get_comment(f"PRSTEP{n}")
    return value, comment


def get_prproc(n: int = 1) -> Tuple[str, str]:
    """
    Get the processing procedure description and standard comment for FITS header.
    This function returns a tuple containing the processing procedure description and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.

    Returns
    -------
    tuple
        A tuple of (processing_procedure_description, standard_comment)
    """
    value = None
    match n:
        case _:
            value = "padre_meddea.calibration.process_file"
    comment = get_comment(f"PRPROC{n}")
    return value, comment


def get_prpver(n: int = 1) -> Tuple[str, str]:
    """
    Get the processing version and standard comment for FITS header.
    This function returns a tuple containing the processing version and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.

    Returns
    -------
    tuple
        A tuple of (processing_version, standard_comment)
    """
    value = padre_meddea.__version__
    comment = get_comment(f"PRPVER{n}")
    return value, comment


def get_prlib(n: int = 1, a: str = "A") -> Tuple[str, str]:
    """
    Get the processing library description and standard comment for FITS header.
    This function returns a tuple containing the processing library description and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.
    a : str, optional
        Library version, default is A.

    Returns
    -------
    tuple
        A tuple of (processing_library_description, standard_comment)
    """
    prlib, _, _ = PRLIBS.get(f"{n}{a}", (None, None, None))
    if prlib:
        return prlib, get_comment(f"PRLIB{n}{a}")
    else:
        raise ValueError(f"Library Undefined for n={n} and a={a}")


def get_prver(n: int = 1, a: str = "A") -> Tuple[str, str]:
    """
    Get the processing version and standard comment for FITS header.
    This function returns a tuple containing the processing version and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.
    a : str, optional
        Library version, default is A.

    Returns
    -------
    tuple
        A tuple of (processing_version, standard_comment)
    """
    _, prver, _ = PRLIBS.get(f"{n}{a}", (None, None, None))
    if prver:
        return prver, get_comment(f"PRVER{n}{a}")
    else:
        raise ValueError(f"Library Undefined for n={n} and a={a}")


def get_prhsh(n: int = 1, a: str = "A") -> Tuple[str, str]:
    """
    Get the processing hash and standard comment for FITS header.
    This function returns a tuple containing the processing hash and
    the corresponding standard comment based on the Processing step.

    Parameters
    ----------
    n : int, optional
        Processing step number, default is 1.
    a : str, optional
        Library version, default is A.

    Returns
    -------
    tuple
        A tuple of (processing_hash, standard_comment)
    """
    lib, version, url = PRLIBS.get(f"{n}{a}", (None, None, None))
    if not url:
        raise ValueError(f"Library Undefined for n={n} and a={a}")

    try:
        # Try Locally
        match a:
            case "A":
                repo = git.Repo(padre_meddea.__file__, search_parent_directories=True)
                hexsha = repo.head.object.hexsha
            case _:
                raise ModuleNotFoundError(f"Library Version Undefined for a={a}")
    except (ValueError, ModuleNotFoundError, git.InvalidGitRepositoryError) as _:
        # Not Available Locally - Use the Remote
        remote_info = git.cmd.Git().ls_remote(url)
        remote_info = remote_info.split()
        # formatted as Dict[tag, hexsha]
        remote_tags = {
            remote_info[i + 1]: remote_info[i] for i in range(0, len(remote_info), 2)
        }
        # Look for Tag
        hexsha = None
        if "dev" not in version:
            version_formattings = [f"v{version}", f"{version}"]
            # Search Various Version Formats
            for version_format in version_formattings:
                hexsha = remote_tags.get(f"refs/tags/{version_format}", None)
                if hexsha is not None:
                    break
        if hexsha is None:
            log.warning(f"Version {version} not found in Tags for {url}. Using HEAD.")
            hexsha = remote_tags.get("HEAD", None)

    # header[f"PRBRA{n}A"] = (
    #    repo.active_branch.name,
    #    get_std_comment(f"PRBRA{n}A"),
    # )
    comment = get_comment(f"PRHSH{n}{a}")
    return hexsha, comment


# =============================================================================
# FITS File Concatenation Functions
# =============================================================================


def _get_output_path(first_file: Path, date_beg: Time) -> Path:
    """
    Determine the output file path if not provided.

    Parameters
    ----------
    first_file : Path
        First file to extract metadata from
    date_beg : Time
        Beginning date for filename generation

    Returns
    -------
    Path
        Determined output file path
    """
    hdul = fits.open(first_file)
    header = hdul[0].header.copy()
    hdul.close()

    instrument = header["INSTRUME"].lower()
    data_type = header["BTYPE"]
    if "_" in data_type:
        data_type = data_type.replace("_", "")

    outfile = create_science_filename(
        instrument, time=date_beg, level="l1", descriptor=data_type, version="0.1.0"
    )

    # Handle temp directory if in Lambda environment
    if os.getenv("LAMBDA_ENVIRONMENT"):
        temp_dir = Path(tempfile.gettempdir())
        outfile = temp_dir / outfile

    return outfile


def _get_combined_list(
    files_to_combine: list[Path], existing_file: Path = None
) -> list[Path]:
    """
    Prepare the list of files to be combined.

    Parameters
    ----------
    files_to_combine : list of Path
        List of FITS files to combine
    existing_file : Path, optional
        Existing daily FITS file to append to

    Returns
    -------
    list of Path
        Sorted list of all files to process
    """
    # Combine with existing file if provided
    all_files = []
    if existing_file:
        all_files.append(existing_file)
    all_files.extend(files_to_combine)

    # Remove duplicates while preserving order
    all_files = list(OrderedDict.fromkeys(all_files))

    return all_files


def _init_hdul_structure(
    template_file: Path,
) -> dict:
    """
    Initialize the HDUL dictionary structure from the first file.

    Parameters
    ----------
    template_file : Path
        First FITS file to use as template
    date_beg, date_end, date_avg : Time
        Calculated time values
    all_parent_files : list[str]
        Combined list of parent files

    Returns
    -------
    dict
        HDU dictionary structure formatted as:
        {
            0: {
                "header": fits.Header,
                "data": None,
                "type": "primary"
            },
            1: {
                "header": fits.Header,
                "data": Table or np.ndarray,
                "type": "bintable" or "image",
                "name": str (optional)
            },
            ...
        }

    """
    hdul_dict = {}

    hdul = fits.open(template_file)
    for i, hdu in enumerate(hdul):
        if isinstance(hdu, fits.PrimaryHDU):
            hdul_dict[i] = {
                "header": hdu.header.copy(),
                "data": None,
                "type": "primary",
                "name": hdu.name,
            }
        elif isinstance(hdu, fits.BinTableHDU):
            if hdu.name in ["PROVENANCE"]:  # Skip non-data HDUs
                continue
            hdul_dict[i] = {
                "header": hdu.header.copy(),
                "data": Table.read(hdu),
                "type": "bintable",
                "name": hdu.name,
            }
        elif isinstance(hdu, fits.ImageHDU):
            hdul_dict[i] = {
                "header": hdu.header.copy(),
                "data": hdu.data.copy(),
                "type": "image",
                "name": hdu.name,
            }
    # Explicitly Open and Close File - Windows Garbage Disposer cannot be trusted.
    hdul.close()

    return hdul_dict


def _concatenate_input_files(input_files: list[Path], hdu_dict: dict) -> dict:
    """
    Process additional files and combine their data with the initial structure.

    Parameters
    ----------
    input_files : list[Path]
        List of files being combined
    hdu_dict : dict
        Initial HDU dictionary structure

    Returns
    -------
    dict
        Updated HDU dictionary with combined data
    """
    for source_file in input_files:
        hdul = fits.open(source_file)
        for i, hdu in enumerate(hdul):
            if i not in hdu_dict:
                log.warning(
                    f"File {source_file} has unexpected HDU at index {i}, skipping."
                )
                continue

            if hdu_dict[i]["type"] == "primary":
                # For primary HDU, we just need to check consistency
                pass
            elif hdu_dict[i]["type"] == "bintable":
                # Vertically stack table data
                new_table = Table.read(hdu)
                hdu_dict[i]["data"] = vstack([hdu_dict[i]["data"], new_table])
            elif hdu_dict[i]["type"] == "image":
                # Concatenate image data (assuming along first axis)
                hdu_dict[i]["data"] = np.concatenate(
                    [hdu_dict[i]["data"], hdu.data], axis=0
                )
        # Explicitly Open and Close File - Windows Garbage Disposer cannot be trusted.
        hdul.close()

    return hdu_dict


def hdu_to_dict(hdu: fits.hdu) -> dict:
    """Given an hdu, convert it to an hdu dict"""

    if isinstance(hdu, fits.BinTableHDU):
        hdu_type = "bintabel"
    elif isinstance(hdu, fits.ImageHDU):
        hdu_type = "image"
    else:
        hdu_type = None

    return {"header": hdu.header, "data": hdu.data, "name": hdu.name, "type": hdu_type}


def get_hdu_data_times(hdu: dict) -> Time:
    """
    Extract time information from the data within a FITS file.

    This function parses times differently based on the file descriptor (eventlist, hk, spec)
    extracted from the filename. It accesses the appropriate HDU and data columns for
    each file type to calculate accurate time values.

    Parameters
    ----------
    hdu : dict
        {
            "header": fits.Header,
            "data": Table or np.ndarray,
            "type": "bintable" or "image",
            "name": str (optional)
        }

    Returns
    -------
    Time
        Astropy Time object containing the time values extracted from the file data

    Raises
    ------
    ValueError
        If the file descriptor is not recognized or supported
    """
    # Get the File Desctiptor
    # We need to parse times differently for Photon / Spectrum / HK
    data_type = hdu["header"].get("BTYPE", "").lower()
    data = hdu["data"]

    # Calculate Times based on the file descriptor
    if data_type == "photon" and hdu["name"] == "SCI":
        times = calc_time(
            data["pkttimes"],
            data["pktclock"],
            data["clocks"],
        )
    elif data_type == "photon" and hdu["name"] == "PKT":
        times = calc_time(data["pkttimes"], data["pktclock"])
    elif data_type == "housekeeping" and hdu["name"] == "HK":
        times = calc_time(data["timestamp"])
    elif data_type == "housekeeping" and hdu["name"] == "READ":
        times = calc_time(data["time_s"], data["time_clock"])
    elif data_type == "spectrum":
        times = calc_time(data["pkttimes"], data["pktclock"])
    else:
        raise ValueError(f"File contents of {hdu['name']} not recogized.")

    return times


def _sort_hdul_template(hdul_dict: dict):

    for i, hdu_info in hdul_dict.items():
        if hdu_info["type"] == "primary":
            # For primary HDU, we just need to check consistency
            pass
        elif hdu_info["type"] == "bintable":
            # Get Times for Bintable Data
            times = get_hdu_data_times(hdu_info)
            # Get sorting indices - this tells you the order that would sort the array
            sort_indices = times.argsort()
            # Replace the data in the HDU with the sorted data
            hdu_info["data"] = hdu_info["data"][sort_indices]
        elif hdu_info["type"] == "image":
            # Sort Image Data
            pass

    return hdul_dict


def split_hdul_by_day(hdul_dict: dict) -> dict:
    """
    Split a FITS HDU dictionary into multiple dictionaries based on day boundaries.

    Parameters:
    -----------
    hdul_dict : dict
        Dictionary representation of a FITS HDU list

    Returns:
    --------
    dict
        Dictionary where keys are days (as strings in 'YYYY-MM-DD' format) and values
        are HDU dictionaries containing only data from that day
    """
    # Get the times from each HDU
    hdu_times = [get_hdu_data_times(hdul_dict[i]) for i in hdul_dict if i != 0]
    unique_times = Time(np.unique(np.concatenate(hdu_times)))

    # Extract the day part of each time
    day_strs = [t.iso[0:10] for t in unique_times]

    # Find unique days
    unique_days = sorted(set(day_strs))

    # Create a dictionary to hold the HDULs for each day
    day_hduls = {}

    # Loop each unique day
    for day in unique_days:
        # Loop each HDU
        for hdu_idx, hdu_info in hdul_dict.items():
            # Store the HDU list for this day
            day_hduls.setdefault(day, {})

            # Create a boolean mask for this day
            hdu_day_strs = [
                t.iso[0:10] for t in hdu_times[hdu_idx - 1]
            ]  # -1 to skip primary HDU
            hud_day_mask = np.array([d == day for d in hdu_day_strs])

            if hdu_info["type"] == "primary":
                # Just copy over the primary HDU header
                day_hduls[day][hdu_idx] = {
                    "header": hdu_info["header"].copy(),
                    "data": None,  # Primary HDU has no data
                    "type": "primary",
                    "name": hdu_info.get("name", None),
                }
            else:
                # Create a copy of the HDU info
                day_hduls[day][hdu_idx] = {
                    "header": hdu_info["header"].copy(),
                    "data": hdu_info["data"][hud_day_mask].copy(),
                    "type": hdu_info["type"],
                    "name": hdu_info.get("name", None),
                }

    return day_hduls


def update_hdul_date_metadata(
    hdul_dict: dict,
) -> dict:
    """
    Function to update the date metadata in the HDU dictionary.
    This function will update the DATE-BEG, DATE-END, DATE-AVG, and DATEREF keywords
    in the headers of the HDUs based on the data contained within them.

    Parameters
    ----------
    hdul_dict : dict
        Dictionary representation of a FITS HDU list, where each key is an index
        and the value is a dictionary with keys "header", "data", "type", and optionally "name".

    Returns
    -------
    dict
        Updated dictionary representation of the FITS HDU list with updated date metadata.
    """

    date_beg = None
    date_end = None

    # Loop through Data HDUs before Primary
    for i, hdu_info in hdul_dict.items():
        log.debug(f"Processing HDU {i} of type {hdu_info['type']}")
        if hdu_info["type"] == "primary":
            # For primary HDU, we just need to check consistency
            continue
        elif hdu_info["type"] == "bintable":
            # Get Times for Bintable Data
            times = get_hdu_data_times(hdu_info)

            # Update Data HDU Header
            if len(times) > 0:
                hdu_info["header"]["DATE-BEG"] = (
                    times[0].fits,
                    get_comment("DATE-BEG"),
                )
                hdu_info["header"]["DATE-END"] = (
                    times[-1].fits,
                    get_comment("DATE-END"),
                )
                hdu_info["header"]["DATE-AVG"] = (
                    (times[0] + (times[-1] - times[0]) / 2).fits,
                    get_comment("DATE-AVG"),
                )
                hdu_info["header"]["DATEREF"] = (times[0].fits, get_comment("DATEREF"))

                # Update info for Primary HDU
                if date_beg is None or times[0] < date_beg:
                    date_beg = times[0]
                if date_end is None or times[-1] > date_end:
                    date_end = times[-1]
            # There are some cases where the data may not have any times
            # I noticed this with the HK CMD_RESP files which did not have times exactly the same in each HDU
            else:
                hdu_info["header"]["DATE-BEG"] = ("", get_comment("DATE-BEG"))
                hdu_info["header"]["DATE-END"] = ("", get_comment("DATE-END"))
                hdu_info["header"]["DATE-AVG"] = ("", get_comment("DATE-AVG"))
                hdu_info["header"]["DATEREF"] = ("", get_comment("DATEREF"))
        elif hdu_info["type"] == "image":
            # Sort Image Data
            pass

    # Update Primary HDU Header with Date/Time Information
    if date_beg is not None and date_end is not None:
        hdul_dict[0]["header"]["DATE-BEG"] = (date_beg.fits, get_comment("DATE-BEG"))
        hdul_dict[0]["header"]["DATE-END"] = (date_end.fits, get_comment("DATE-END"))
        hdul_dict[0]["header"]["DATE-AVG"] = (
            (date_beg + (date_end - date_beg) / 2).fits,
            get_comment("DATE-AVG"),
        )
        hdul_dict[0]["header"]["DATEREF"] = (date_beg.fits, get_comment("DATEREF"))

    return hdul_dict


def update_hdul_filename_metadata(
    hdul_dict: dict,
    output_file: Path,
    provenance_table: Table = None,
) -> dict:
    """
    Function to update the filename metadata in the HDU dictionary.
    This function will update the FILENAME keyword in the headers of the HDUs
    based on the output file name.

    Parameters
    ----------
    hdul_dict : dict
        Dictionary representation of a FITS HDU list, where each key is an index
        and the value is a dictionary with keys "header", "data", "type", and optionally "name".
    output_file : Path
        path the hdul_dict is being written to, used to extract filename metadata.

    Returns
    -------
    dict
        Updated dictionary representation of the FITS HDU list with updated filename metadata.
    """
    filename_meta = parse_science_filename(output_file)
    # Update Filename
    hdul_dict[0]["header"]["FILENAME"] = (output_file, get_comment("FILENAME"))
    hdul_dict[0]["header"]["LEVEL"] = (filename_meta["level"], get_comment("LEVEL"))

    # Generate the PARENTXT string from the provenance table filenames
    if provenance_table is not None:
        parent_files = provenance_table["FILENAME"].tolist()
        parent_files_str = ", ".join(parent_files)
        hdul_dict[0]["header"]["PARENTXT"] = (
            parent_files_str,
            "Parent files used in concatenation",
        )
    else:
        # If no provenance table, just set PARENTXT to the output file
        hdul_dict[0]["header"]["PARENTXT"] = (
            str(output_file),
            "Parent file used in concatenation",
        )

    return hdul_dict


def split_provenance_tables_by_day(files_to_combine, existing_file=None):
    """
    Splits provenance entries into daily Astropy Tables. If a file spans
    multiple days, it is duplicated across those days with start/end times clipped
    to each day. Take note if the file has no DATE-BEG or DATE-END in the header,
    it will utilize the file DATEREF as a fallback.

    Parameters
    ----------
    files_to_combine : list[Path or object with .name]
        List of FITS file paths or similar objects.
    existing_file : Path or None
        Optional existing provenance FITS file to pull previous provenance from.

    Returns
    -------
    dict[str, astropy.table.Table]
        Dictionary mapping 'YYYY-MM-DD' -> Table with columns FILENAME, DATE_BEG, DATE_END
    """
    by_day = defaultdict(list)

    # STEP 1: Load provenance from existing file if it exists
    if existing_file and Path(existing_file).exists():
        with fits.open(existing_file) as hdul:
            try:
                existing_table = Table(hdul["PROVENANCE"].data)
                for row in existing_table:
                    day = row["DATE_BEG"][:10]  # 'YYYY-MM-DD'
                    by_day[day].append(
                        {
                            "FILENAME": row["FILENAME"],
                            "DATE_BEG": row["DATE_BEG"],
                            "DATE_END": row["DATE_END"],
                        }
                    )
            except (KeyError, AttributeError):
                pass  # skip if PROVENANCE not present or invalid

    # STEP 2: Add new provenance entries
    for fileobj in files_to_combine:
        filename = fileobj.name if hasattr(fileobj, "name") else Path(fileobj).name

        with fits.open(fileobj) as hdul:
            hdr = hdul[0].header

            start_str = hdr.get("DATE-BEG")
            end_str = hdr.get("DATE-END")
            fallback = hdr.get("DATEREF")

            if not start_str or not end_str:
                if fallback:
                    start_str = end_str = fallback
                else:
                    raise ValueError(
                        f"Missing DATE-BEG or DATE-END in header for {filename}"
                    )

            start = Time(start_str, format="fits", scale="utc")
            end = Time(end_str, format="fits", scale="utc")

        start_day = datetime.strptime(start.utc.iso[:10], "%Y-%m-%d")
        end_day = datetime.strptime(end.utc.iso[:10], "%Y-%m-%d")

        day = start_day
        while day <= end_day:
            day_start = Time(
                datetime.combine(day, datetime.min.time()),
                format="datetime",
                scale="utc",
            )
            day_end = Time(
                datetime.combine(day, time(23, 59, 59, 999000)),
                format="datetime",
                scale="utc",
            )

            clipped_start = max(start, day_start)
            clipped_end = min(end, day_end)

            by_day[day.strftime("%Y-%m-%d")].append(
                {
                    "FILENAME": filename,
                    "DATE_BEG": clipped_start.fits,
                    "DATE_END": clipped_end.fits,
                }
            )

            day += timedelta(days=1)

    # STEP 3: Convert to sorted Astropy tables
    tables_by_day = {}
    for day, entries in by_day.items():
        entries.sort(key=lambda e: Time(e["DATE_BEG"]).mjd)
        table = Table(rows=entries, names=["FILENAME", "DATE_BEG", "DATE_END"])
        tables_by_day[day] = table

    return tables_by_day


def _write_output_file(
    hdu_dict: dict, outfile: Path, retries: int = 5, delay: float = 1.0
) -> None:
    """
    Construct and write the output FITS file with retry mechanism.

    Parameters
    ----------
    hdu_dict : dict
        Dictionary containing HDU information
    outfile : Path
        Output file path
    retries : int, optional
        Number of retry attempts, default is 3
    delay : float, optional
        Delay between retries in seconds, default is 1.0
    """
    hdu_list = []
    for i in sorted(hdu_dict.keys()):
        hdu_info = hdu_dict[i]

        if hdu_info["type"] == "primary":
            hdu_list.append(fits.PrimaryHDU(header=hdu_info["header"]))
        elif hdu_info["type"] == "bintable":
            new_hdu = fits.BinTableHDU(
                hdu_info["data"], header=hdu_info["header"], name=hdu_info["name"]
            )
            new_hdu.add_checksum()
            hdu_list.append(new_hdu)
        elif hdu_info["type"] == "image":
            new_hdu = fits.ImageHDU(
                hdu_info["data"], header=hdu_info["header"], name=hdu_info["name"]
            )
            hdu_list.append(new_hdu)

    # Write the output file with retry mechanism
    hdul = fits.HDUList(hdu_list)
    attempt = 0
    while attempt < retries:
        try:
            log.debug(f"Writing to file: {outfile} (Attempt {attempt + 1})")
            hdul.writeto(outfile, overwrite=True)
            log.info(f"Created concatenated daily file: {outfile}")
            return outfile
        except Exception as e:
            attempt += 1
            log.warning(f"Failed to write file {outfile} on attempt {attempt}: {e}")
            if attempt < retries:
                gc.collect()  # Explicitly invoke garbage collection to release any unreferenced file handles
                time.sleep(delay)
            else:
                log.error(f"Exceeded maximum retries ({retries}) for file {outfile}")
                raise
        finally:
            log.debug(f"Closing file: {outfile}")
            hdul.close()


def concatenate_files(
    files_to_combine: list[Path],
    existing_file: Path = None,
) -> list[Path]:
    """
    Concatenate multiple FITS files into a single daily FITS file, properly combining headers and data.

    The function also tracks all parent files in the PARENTXT header keyword.

    Note: This function assumes that there is no data stored in the PrimaryHDU of the FITS files.

    Parameters
    ----------
    files_to_combine : list of Path
        List of FITS files to combine. Assumed to have the same structure.
    existing_file : Path, optional
        Existing daily FITS file to append to.

    Returns
    -------
    output_file : list of Path
        List containing the output file paths of the concatenated daily FITS files.
    """
    # Ignore MergeConflictWarning from astropy
    warnings.simplefilter("ignore", MergeConflictWarning)

    # Get the combined list of files to process
    all_files = _get_combined_list(files_to_combine, existing_file)

    # Create new provenance table from the files to combine
    provenance_tables = split_provenance_tables_by_day(files_to_combine, existing_file)

    # Initialize Data Structures
    hdul_dict = _init_hdul_structure(all_files[0])

    # Concatenate Input Files
    hdul_dict = _concatenate_input_files(all_files[1:], hdul_dict)

    # Sort Data Structures by Time
    hdul_dict = _sort_hdul_template(hdul_dict)

    # Split HDU by Day
    hdul_dicts = split_hdul_by_day(hdul_dict)

    outfiles = []
    # Save each Day
    for day, day_hdul in hdul_dicts.items():

        # Calculate the Outputn Path Filename
        outfile = _get_output_path(
            first_file=all_files[0], date_beg=Time(day + "T00:00:00")
        )

        # Update HDUL Primary Header with Date/Time Information
        day_hdul = update_hdul_date_metadata(day_hdul)

        log.info(f"Processing day: {day} with output file: {outfile}")
        log.info(f"Provenance Table: {provenance_tables[day]}")

        # Update HDUL Primary Header with Filename Information
        day_hdul = update_hdul_filename_metadata(
            day_hdul, outfile, provenance_tables[day]
        )

        # Add Provenance Table to HDU
        if day in provenance_tables:
            prov_data = provenance_tables[day]

            prov_table = {
                "header": fits.Header(
                    [
                        ("EXTNAME", "PROVENANCE", get_comment("EXTNAME")),
                        (
                            "COMMENT",
                            "Provenance information for the concatenated files",
                        ),
                        ("OBS_HDU", 0, get_comment("OBS_HDU")),
                    ]
                ),
                "data": prov_data,
                "type": "bintable",
                "name": "PROVENANCE",
            }
            day_hdul[max(day_hdul) + 1] = prov_table

        # Write output file
        out_path = _write_output_file(day_hdul, outfile)

        outfiles.append(Path(out_path))

    # This should return the list of Paths of `outfiles` that were successfully created,
    return outfiles
