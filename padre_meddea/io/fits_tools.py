"""
This module provides a utilities to manage fits files reading and writing.
"""

import re
import git
from git import InvalidGitRepositoryError

from astropy.io import ascii
import astropy.io.fits as fits
from astropy.time import Time

import padre_meddea

FITS_HDR0 = ascii.read(
    padre_meddea._data_directory / "fits" / "fits_keywords_primaryhdu.csv", format="csv"
)
FITS_HDR0.add_index("keyword")
FITS_HDR_KEYTOCOMMENT = ascii.read(
    padre_meddea._data_directory / "fits" / "fits_keywords_dict.csv", format="csv"
)
FITS_HDR_KEYTOCOMMENT.add_index("keyword")


def get_std_comment(keyword: str) -> str:
    """Given a keyword, return the standard comment for a header card."""
    if keyword.upper() in FITS_HDR_KEYTOCOMMENT["keyword"]:
        return FITS_HDR_KEYTOCOMMENT.loc[keyword]["comment"]
    for this_row in FITS_HDR_KEYTOCOMMENT:
        res = re.fullmatch(this_row["keyword"], keyword)
        if res:
            comment = this_row["comment"]
            if len(res.groupdict()) > 0:  # check if there was a match
                for key, value in res.groupdict().items():
                    comment = comment.replace(f"<{key}>", value)
            return comment


def get_primary_header() -> fits.Header:
    """Return a standard FITS file primary header."""
    header = fits.Header()
    header["DATE"] = (Time.now().fits, get_std_comment("DATE"))
    for row in FITS_HDR0:
        this_comment = get_std_comment(row["keyword"])
        header[row["keyword"]] = (row["value"], this_comment)
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
    except InvalidGitRepositoryError:
        pass
    #  primary_hdr["PRLOG1"] add log information, need to do this after the fact
    #  primary_hdr["PRENV1"] add information about processing env, need to do this after the fact
    return header
