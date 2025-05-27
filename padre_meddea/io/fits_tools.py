"""
This module provides a utilities to manage fits files reading and writing.
"""

from collections import OrderedDict
import os
from pathlib import Path
import re
from typing import List, Tuple
import json
import warnings

import git
import ccsdspy
from astropy import units as u
from astropy.io import ascii
import astropy.io.fits as fits
from astropy.time import Time
from astropy.table import Table, vstack
from astropy.utils.metadata import MergeConflictWarning
import numpy as np

import solarnet_metadata
from solarnet_metadata.schema import SOLARNETSchema


import padre_meddea
from padre_meddea import log
from padre_meddea.util.util import create_science_filename, calc_time

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
    header["BUNIT"] = get_bunit(data_level, data_type)

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


def get_obs_header():
    """
    Create a standard FITS header for the observation.

    This function creates a new FITS header with standard metadata including the
    current date, default PADRE attributes, processing information, data level,
    data type, original APID, and original filename.

    Returns
    -------
    fits.Header
        A FITS header populated with standard metadata
    """
    # Create a new header
    header = fits.Header()
    header["OBS_HDU"] = (1, get_comment("OBS_HDU"))
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


def concatenate_daily_fits(
    files_to_combine: list[Path], existing_file: Path = None, outfile: Path = None
) -> Path:
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
    outfile : Path, optional
        Output file path. If None, will generate automatically.

    Returns
    -------
    output_file : Path
        Path to the concatenated daily FITS file.
    """

    # Ignore MergeConflictWarning from astropy
    warnings.simplefilter("ignore", MergeConflictWarning)

    # Combine with existing file if provided
    all_files = []
    if existing_file:
        all_files.append(existing_file)
    all_files.extend(files_to_combine)

    # Remove duplicates while preserving order
    all_files = list(OrderedDict.fromkeys(all_files))

    # Sort files by observation time
    all_files = sorted(all_files, key=lambda f: fits.getheader(f)["DATE-BEG"])

    # Calculate time range for the output file
    all_times = []
    for fits_file in all_files:
        hdr = fits.getheader(fits_file)
        for key in ("DATE-BEG", "DATE-END"):
            if key in hdr:
                all_times.append(Time(hdr[key]))

    date_beg = Time(min(all_times).iso[0:10])
    date_end = max(all_times)
    date_avg = date_beg + (date_end - date_beg) / 2

    # Determine output path if not provided
    if outfile is None:
        instrument = fits.getheader(all_files[0])["INSTRUME"].lower()
        data_type = fits.getheader(all_files[0])["DATATYPE"]
        if "_" in data_type:
            data_type = data_type.replace("_", "")

        outfile = create_science_filename(
            instrument, time=date_beg, level="l1", descriptor=data_type, version="0.1.0"
        )

        # Handle temp directory if in Lambda environment
        if os.getenv("LAMBDA_ENVIRONMENT"):
            temp_dir = Path(tempfile.gettempdir())
            outfile = temp_dir / outfile

    # Prepare parent file tracking
    new_parent_files = []
    existing_parent_files = []

    # Collect new files being added (excluding existing_file)
    for file_path in files_to_combine:
        new_parent_files.append(str(file_path.name))

    # Initialize the HDU structure from the first file
    with fits.open(all_files[0]) as first_hdul:
        hdu_dict = {}

        # Process each HDU in the first file
        for i, hdu in enumerate(first_hdul):
            if isinstance(hdu, fits.PrimaryHDU):
                # Start with the primary header
                base_header = hdu.header.copy()

                # Update time-related headers
                for key, value in [
                    ("DATE-BEG", date_beg),
                    ("DATE-END", date_end),
                    ("DATE-AVG", date_avg),
                    ("DATEREF", date_beg),
                ]:
                    base_header[key] = (value.fits, get_comment(key))

                # PARENTXT is a comma-separated list of parent files
                # Extract existing PARENTXT if this is an existing concatenated file
                if existing_file and "PARENTXT" in base_header:
                    existing_parent_files = [
                        f.strip() for f in base_header["PARENTXT"].split(",")
                    ]
                # Update PARENTXT with all parent files
                all_parent_files = existing_parent_files + new_parent_files

                # Remove duplicates while preserving order
                unique_parent_files = list(OrderedDict.fromkeys(all_parent_files))

                parent_files_str = ", ".join(unique_parent_files)
                base_header["PARENTXT"] = (
                    parent_files_str,
                    "Parent files used in concatenation",
                )
                # Build or update COMMENT JSON metadata withe file names and times
                comment_raw = base_header.get("COMMENT", "")

                # COMMENT may be a list of strings (FITS header lines are <=72 chars)
                if isinstance(comment_raw, list):
                    comment_str = "".join(comment_raw)  # Avoid actual newlines
                else:
                    comment_str = str(comment_raw).replace("\n", "")

                file_time_list = []
                try:
                    file_time_list = json.loads(comment_str)
                except json.JSONDecodeError as e:
                    log.warning(f"Failed to parse COMMENT as JSON: {e}")

                # Add new metadata entries (avoiding duplicates by filename)
                existing_filenames = {entry["filename"] for entry in file_time_list}
                for file_path in files_to_combine:
                    filename = file_path.name
                    if filename not in existing_filenames:
                        hdr = fits.getheader(file_path)
                        file_time_list.append(
                            {
                                "filename": filename,
                                "date-beg": hdr.get("DATE-BEG", "UNKNOWN"),
                                "date-end": hdr.get("DATE-END", "UNKNOWN"),
                            }
                        )
                        existing_filenames.add(filename)

                # Sort by date-beg using sorted
                file_time_list = sorted(
                    file_time_list,
                    key=lambda x: (
                        Time(x.get("date-beg", "UNKNOWN")).mjd
                        if x.get("date-beg") != "UNKNOWN"
                        else float("inf")
                    ),
                )

                # Store updated JSON as string in header
                json_str = json.dumps(file_time_list, separators=(",", ":"))

                # Remove existing COMMENT keyword if present to avoid duplicates
                while "COMMENT" in base_header:
                    base_header.remove("COMMENT")
                base_header["COMMENT"] = (json_str, "JSON list of contributing files")

                hdu_dict[i] = {"header": base_header, "data": None, "type": "primary"}

            elif isinstance(hdu, fits.BinTableHDU):
                hdu_dict[i] = {
                    "header": hdu.header.copy(),
                    "data": Table.read(hdu),
                    "type": "bintable",
                    "name": hdu.name,
                }
            elif isinstance(hdu, fits.ImageHDU):
                hdu_dict[i] = {
                    "header": hdu.header.copy(),
                    "data": hdu.data.copy(),
                    "type": "image",
                    "name": hdu.name,
                }

    # Process additional files
    for file_path in all_files[1:]:
        with fits.open(file_path) as hdul:
            for i, hdu in enumerate(hdul):
                if i not in hdu_dict:
                    log.warning(
                        f"File {file_path} has unexpected HDU at index {i}, skipping."
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

    # Construct the output HDUList
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

    # Write the output file
    hdul = fits.HDUList(hdu_list)
    hdul.writeto(outfile, overwrite=True)
    log.info(f"Created concatenated daily file: {outfile}")

    return Path(outfile)


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
