"""
Provides general utility functions.
"""

import csv
import os
import re
import tempfile
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries
from ccsdspy.utils import split_by_apid, split_packet_bytes
from sunpy.net.attr import AttrAnd
from swxsoc.util import (
    Descriptor,
    Instrument,
    Level,
    SearchTime,
    SWXSOCClient,
    create_science_filename,
    parse_science_filename,
)

import padre_meddea
from padre_meddea import APID, EPOCH, log

# used to identify bad times
MIN_TIME_BAD = Time("2024-02-01T00:00")

__all__ = [
    "parse_science_filename",
    "create_science_filename",
    "create_meddea_filename",
    "get_filename_version_base",
    "increment_filename_version",
    "calc_time",
    "has_baseline",
    "is_consecutive",
]


def create_meddea_filename(
    time: Time,
    level: str,
    descriptor: str,
    test: str,
    overwrite: bool = False,
) -> str:
    """
    Generate the MEDDEA filename based on the provided parameters.

    Parameters
    ----------
    time : Time
        The time associated with the data.
    level : str
        The data level (e.g., "L1", "L2").
    descriptor : str
        The data descriptor (e.g., "SCI", "CAL").
    test : str
        The test identifier (e.g., "TEST1", "TEST2").
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    str
        The generated MEDDEA filename.
    """
    # Filename Version X.Y.Z comes from two parts:
    #   1. Files Version Base: X.Y comes from the Software Version -> Data Version Mapping
    #   2. File Version Incrementor: Z starts at 0 and iterates for each new version based on what already exists in the filesystem.
    version_base = get_filename_version_base()
    version_increment = 0
    version_str = f"{version_base}.{version_increment}"

    # The Base Filename is used for searching to see if we need to increase our version increment.
    base_filename = create_science_filename(
        instrument="meddea",
        time=time,
        level=level,
        descriptor=descriptor,
        test=test,
        version=version_str,
    )

    # Check if the LAMBDA_ENVIRONMENT environment variable is set
    lambda_environment = os.getenv("LAMBDA_ENVIRONMENT")
    # If we just want to overwrite existing files, the don't bother checking if the version exists
    if overwrite:
        if lambda_environment:
            temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
            output_path = temp_dir / base_filename
        else:
            output_path = Path(base_filename)
        # Return early
        return output_path

    # check if file already exists, if it exists set version to x.y.(max(z)+1)
    search_pattern = base_filename.replace(version_str, f"{version_str[0:-1]}*")
    if lambda_environment:
        # search for existing file in AWS for all files with x.y.z choose largest z and set to x.y.z+1
        # Convert the glob pattern to regex pattern
        regex_pattern = search_pattern.replace(".", "\\.").replace("*", ".*")
        regex = re.compile(regex_pattern)
        # Search Lambda Environment for Files
        client = SWXSOCClient()
        try:
            results = client.search(
                AttrAnd(
                    [
                        SearchTime(start=time, end=time),
                        Instrument("meddea"),
                        Level(level),
                        Descriptor(descriptor),
                    ]
                )
            )
            # Find matches
            matching_files = [
                Path(result["key"]).name
                for result in results
                if regex.match(Path(result["key"]).name)
            ]
        except Exception as e:
            log.error(f"Error Searching for Files in Lambda Environment: {e}")
            matching_files = []
        # Check if there are any matching files, if so we need to increment.
        if len(matching_files) > 0:
            existing_versions = [
                int(parse_science_filename(this_f)["version"].split(".")[-1])
                for this_f in matching_files
            ]
            incremented_filename = create_science_filename(
                "meddea",
                time=time,
                level=level,
                descriptor=descriptor,
                test=test,
                version=f"{version_base}.{max(existing_versions) + 1}",
            )
        else:
            incremented_filename = base_filename
        # Andrew insert code here
        temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
        output_path = temp_dir / incremented_filename
    else:
        # Search if File Exists Locally
        if Path(base_filename).exists():
            existing_files = Path.cwd().glob(search_pattern)
            existing_versions = [
                int(parse_science_filename(this_f)["version"].split(".")[-1])
                for this_f in existing_files
            ]
            incremented_filename = create_science_filename(
                "meddea",
                time=time,
                level=level,
                descriptor=descriptor,
                test=test,
                version=f"{version_base}.{max(existing_versions) + 1}",
            )
        else:
            incremented_filename = base_filename
        # Return a Path with the local incremented Filename
        output_path = Path(incremented_filename)

    return output_path


def get_filename_version_base() -> str:
    """
    Get the two most significant bits of the version number based on the current version of the software.

    Returns
    -------
    str
        The base version string for the filename. For example, "1.0".
    """
    version_mapping_path = (
        padre_meddea._data_directory / "software_to_data_version_mapping.csv"
    )

    # Read the version mapping CSV
    version_mapping = {}
    try:
        with open(version_mapping_path, "r") as f:
            csv_reader = csv.reader(f)

            for row in csv_reader:
                if not row:  # Skip empty rows
                    continue

                software_version = row[0]  # First column value is the key
                data_version = row[1]  # Rest of the row as values

                if software_version in version_mapping:
                    version_mapping[software_version].append(data_version)
                else:
                    version_mapping[software_version] = [data_version]
    except Exception as e:
        log.error(f"Error reading version mapping file: {e}")
        version_mapping = {}  # Empty dictionary in case of error

    # Sort the Data Versions for each Software Version - Then we can get the latest version from Index 0
    for key in version_mapping.keys():
        version_mapping[key].sort(
            reverse=True
        )  # Sort in descending order to get the latest version first

    # Get the Latest Version (based on keys in the mapping)
    latest_version = max(version_mapping.keys())

    # Get the two most significant bits of the current version number
    meddea_version = padre_meddea.__version__
    current_version_key = ".".join(
        meddea_version.split(".")[:2]
    )  # Get the first two parts of the version

    if current_version_key in version_mapping:
        # Return the latest data version for the current software version
        return version_mapping[current_version_key][0]
    else:
        log.warning(
            f"No data version found for software version {current_version_key}. Defaulting to Latest Version. ({latest_version})"
        )
        return version_mapping[latest_version][0]


def increment_filename_version(file_path: Path, version_index=0):
    """Given a filename, increment the version number by one.

    Parameter
    ---------
    version_index: int
        The version index to increment. Index 0 is least significant version.

    Returns
    -------
    filename : str
    """
    file_path = Path(file_path)
    tokens = parse_science_filename(file_path.name)
    version_tuple = [int(i) for i in tokens["version"].split(".")]
    version_tuple.reverse()
    version_tuple[version_index] += 1
    version_str = f"{version_tuple[2]}.{version_tuple[1]}.{version_tuple[0]}"
    return file_path.with_name(file_path.name.replace(tokens["version"], version_str))


def calc_time(pkt_time_s, pkt_time_clk=0, ph_clk=0) -> Time:
    """
    Convert times to a Time object
    """
    deltat = TimeDelta(
        pkt_time_s * u.s
        + pkt_time_clk * 0.05 * u.microsecond
        + ph_clk * 12.8 * u.microsecond
    )
    result = Time(EPOCH + deltat)
    return result


def has_baseline(filename: Path, packet_count=10) -> bool:
    """Given a stream of photon packets, check whether the baseline measurement is included.
    Baseline packets have one extra word per photon for a total of 4 words (8 bytes).

    This function calculates the number of hits in the packet assuming 4 words per photon.
    If the resultant is not an integer number then returns False.

    Parameters
    ----------
    packet_bytes : byte string
        Photon packet bytes, must be an integer number of whole packets and greaterh

    Returns
    -------
    result : bool

    """
    HEADER_BYTES = 11 * 16 / 8
    BYTES_PER_PHOTON = 16 * 4 / 8

    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
        if APID["photon"] in stream_by_apid.keys():  # only applicable to photon packets
            packet_stream = stream_by_apid[APID["photon"]]
            packet_bytes = split_packet_bytes(packet_stream)
            packet_count = min(
                len(packet_bytes), packet_count
            )  # in case we have fewer than packet_count in the file
            num_hits = np.zeros(packet_count)
            for i in range(packet_count):
                num_hits[i] = (len(packet_bytes[i]) - HEADER_BYTES) / BYTES_PER_PHOTON
        else:
            raise ValueError("Only works on photon packets.")
    # check if there is any remainder for non integer number of hits
    return np.sum(num_hits - np.floor(num_hits)) == 0


def str_to_fits_keyword(keyword: str) -> str:
    """Given a keyword string, return a fits compatible keyword string
    which must not include special characters and have fewer have no more
    than 8 characters."""
    clean_keyword = "".join(e for e in keyword if e.isalnum()).strip().upper()
    return clean_keyword[0:8]


def is_consecutive(arr: np.array) -> bool:
    """Return True if the packet sequence numbers are all consecutive integers, has no missing numbers."""
    MAX_SEQCOUNT = 2**14 - 1  # 16383

    # Ensure arr is at least 1D
    arr = np.atleast_1d(arr)

    # check if seqcount has wrapped around
    indices = np.where(arr == MAX_SEQCOUNT)
    if len(indices[0]) == 0:  # no wrap
        return np.all(np.diff(arr) == 1)
    else:
        last_index = 0
        result = True
        for this_ind in indices[0]:
            this_arr = arr[last_index : this_ind + 1]
            result = result & np.all(np.diff(this_arr) == 1)
            last_index = this_ind + 1
        # now do the remaining part of the array
        this_arr = arr[last_index + 1 :]
        result = result & np.all(np.diff(this_arr) == 1)
        return result


def trim_timeseries(ts: TimeSeries, t0=MIN_TIME_BAD) -> TimeSeries:
    """Remove all times in a time series before a given time.
    Parameters
    ----------
    ts : Time
        The time before which all data should be removed.

    Returns
    -------
    TimeSeries
        A TimeSeries containing the trimmed time series.
    """
    inds = ts.time < t0
    bad_count = np.sum(inds)
    if bad_count > 0:
        log.warning(f"Found {bad_count} bad times. Removing them.")
    return ts[~inds]


def parse_ph_flags(ph_flags):
    """Given the photon flag field, parse into its individual components.
    The flags are stored as follows
    [15] Int.Time Overflow, [14:12] decimation level, [11:0] # dropped photons

    Returns
    -------
    decim_lvl, dropped_count, int_time_overflow
    """
    decim_lvl = (ph_flags >> 12) & 0b0111
    dropped_count = ph_flags & 2047
    int_time_overflow = ph_flags & 32768
    return decim_lvl, dropped_count, int_time_overflow


def threshold_to_energy(threshold_value: int) -> u.Quantity:
    """Given a threshold value return the threshold energy.
    If given a threshold of 63, the pixel is disabled.
        In that case, return the maximum detectable energy of 100 keV
    Parameters
    ----------
    threshold_value : int
        The threshold value
    Returns
    -------
    threshold_energy : u.Quantity
        The energy value of the threshold.
    """
    MAX_ENERGY = 100 * u.keV
    max_threshold = 63
    if (0 > threshold_value) or (threshold_value > max_threshold):
        raise ValueError("Threshold value should be between 0 and 63.")
    if threshold_value == 63:
        return MAX_ENERGY
    msb = 0.8 * u.keV
    lsb = 0.2 * u.keV
    lsb_array = np.arange(-3, 60, 1)
    lsb_array[lsb_array > 52] = 52
    msb_array = np.zeros(max_threshold)
    msb_array[-8:] = np.arange(0, 8)
    threshold_energy = msb_array * msb + lsb_array * lsb
    return threshold_energy[threshold_value]


def get_file_time(filename) -> Time:
    """Given filename return the time stamp."""
    return Time(f"{filename[0:4]}-{filename[4:6]}-{filename[6:8]}T00:00")
