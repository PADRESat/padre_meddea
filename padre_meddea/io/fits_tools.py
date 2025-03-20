"""
This module provides a utilities to manage fits files reading and writing.
"""

from pathlib import Path
import re
from typing import List, Tuple

from astropy.io import ascii
import astropy.io.fits as fits
from astropy.time import Time
from solarnet_metadata.schema import SOLARNETSchema

import padre_meddea

CUSTOM_ATTRS_PATH = (
    padre_meddea._data_directory / "fits" / "fits_keywords_primaryhdu.yaml"
)

FITS_HDR_KEYTOCOMMENT = ascii.read(
    padre_meddea._data_directory / "fits" / "fits_keywords_dict.csv", format="csv"
)
FITS_HDR_KEYTOCOMMENT.add_index("keyword")


def get_std_comment(keyword: str) -> str:
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
    header["DATE"] = (Time.now().fits, get_std_comment("DATE"))

    # Add PADRE Default Attributes to Header
    for keyword, value in schema.default_attributes.items():
        header[keyword] = (value, get_std_comment(keyword))

    # Add PADRE Git Information to Header
    header = add_process_info_to_header(header, n=procesing_step)

    # Add Data Level
    header["LEVEL"] = (data_level, get_std_comment("LEVEL"))

    # Add Data Type
    header["DATATYPE"] = (data_type, get_std_comment("DATATYPE"))

    # Add Original APID
    header["ORIGAPID"] = (
        padre_meddea.APID[data_type],
        get_std_comment("ORIGAPID"),
    )

    # Add Original File Name
    file_path = Path(file_path)
    header["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))

    return header


def validate_fits_header(header: fits.Header) -> List[str]:
    """Validate a fits header against the SOLARNET schema.

    Parameters
    ----------
    header : fits.Header
        The fits header to validate

    Returns
    -------
    errors : List[str]
        A list of errors found during validation
    """
    warning = "Header Validation Finding:"
    # Create a Custom SOLARNET Schema
    schema = SOLARNETSchema(schema_layers=[CUSTOM_ATTRS_PATH])

    header_dict = {}
    for card in header.cards:
        header_dict[card.keyword] = (card.value, card.comment)
    # Run Validation
    validation_findings = schema.validate(header_dict)

    return validation_findings


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

    header[f"PRSTEP{n}"] = get_prstep(n)
    header[f"PRPROC{n}"] = get_prproc(n)
    header[f"PRPVER{n}"] = get_prpver(n)
    header[f"PRLIB{n}A"] = get_prlib(n, "A")
    header[f"PRVER{n}A"] = get_prver(n, "A")
    header[f"PRHSH{n}A"] = get_prhsh(n, "A")
    # header[f"PRBRA{n}A"] = get_prbra(n, "A")

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
    comment = get_std_comment(f"PRSTEP{n}")
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
    comment = get_std_comment(f"PRPROC{n}")
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
    comment = get_std_comment(f"PRPVER{n}")
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
    match a:
        case "A":
            value = "padre_meddea"
        case _:
            raise ValueError(f"Library Version Undefined for a={a}")

    comment = get_std_comment(f"PRLIB{n}{a}")
    return value, comment


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
    match a:
        case "A":
            value = padre_meddea.__version__
        case _:
            raise ValueError(f"Library Version Undefined for a={a}")

    comment = get_std_comment(f"PRVER{n}{a}")
    return value, comment


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
    try:
        import git
        from git import InvalidGitRepositoryError

        match a:
            case "A":
                repo = git.Repo(padre_meddea.__file__, search_parent_directories=True)
                value = repo.head.object.hexsha
            case _:
                raise ValueError(f"Library Version Undefined for a={a}")

        comment = get_std_comment(f"PRHSH{n}{a}")
    except ModuleNotFoundError:
        value = None
        comment = None
    except InvalidGitRepositoryError:
        value = None
        comment = None

    # header[f"PRBRA{n}A"] = (
    #    repo.active_branch.name,
    #    get_std_comment(f"PRBRA{n}A"),
    # )
    # commits = list(repo.iter_commits("main", max_count=1))
    # header[f"PRVER{n}B"] = (
    #    Time(commits[0].committed_datetime).fits,
    #    get_std_comment(f"PRVER{n}B"),
    # )
    return value, comment
