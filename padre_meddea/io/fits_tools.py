"""
This module provides a utilities to manage fits files reading and writing.
"""

from pathlib import Path
import re

from astropy.io import ascii
import astropy.io.fits as fits
from astropy.time import Time
from astropy.table import Table, vstack

import padre_meddea
from padre_meddea import log

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


def concatenate_daily_fits(
    files_to_combine: list[Path], existing_file: Path = None, outfile: Path = None
) -> Path:
    """
    Concatenate multiple FITS files into a single daily FITS file, properly combining headers and data.

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

    # Combine with existing file if provided
    all_files = []
    if existing_file:
        all_files.append(existing_file)
    all_files.extend(files_to_combine)

    # Remove duplicates while preserving order
    seen = set()
    all_files = [f for f in all_files if not (f in seen or seen.add(f))]

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
                    base_header[key] = (value.fits, get_std_comment(key))

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

    return outfile
