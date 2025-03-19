"""
This module provides a utilities to manage fits files reading and writing.
"""

from pathlib import Path
import re

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
        Data processing level (e.g., 'L0', 'L1')
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


def add_process_info_to_header(header: fits.Header, n=1) -> fits.Header:
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
    if n < 1:
        ValueError("Processing number, n, must be greater than or equal to 1.")
    header[f"PRSTEP{n}"] = ("PROCESS Raw to L1", get_std_comment(f"PRSTEP{n}"))
    header[f"PRPROC{n}"] = (
        "padre_meddea.calibration.process",
        get_std_comment(f"PRPROC{n}"),
    )
    header[f"PRPVER{n}"] = (
        padre_meddea.__version__,
        get_std_comment(f"PRPVER{n}"),
    )
    header[f"PRLIB{n}A"] = (
        "padre_meddea",
        get_std_comment(f"PRLIB{n}A"),
    )
    header[f"PRVER{n}A"] = (padre_meddea.__version__, get_std_comment(f"PRVER{n}A"))
    try:
        import git
        from git import InvalidGitRepositoryError

        repo = git.Repo(padre_meddea.__file__, search_parent_directories=True)
        header[f"PRHSH{n}A"] = (
            repo.head.object.hexsha,
            get_std_comment(f"PRHSH{n}A"),
        )
        # header[f"PRBRA{n}A"] = (
        #    repo.active_branch.name,
        #    get_std_comment(f"PRBRA{n}A"),
        # )
        # commits = list(repo.iter_commits("main", max_count=1))
        # header[f"PRVER{n}B"] = (
        #    Time(commits[0].committed_datetime).fits,
        #    get_std_comment(f"PRVER{n}B"),
        # )
    except ModuleNotFoundError:
        pass
    except InvalidGitRepositoryError:
        pass
    #  primary_hdr["PRLOG1"] add log information, need to do this after the fact
    #  primary_hdr["PRENV1"] add information about processing env, need to do this after the fact
    return header
