"""
Provides tools for the PADRE MeDDEA flare list.
"""

import csv
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.table import MaskedColumn
from astropy.time import Time
from astropy.timeseries import TimeSeries


def read_flare_list(filename: Path) -> TimeSeries:
    """Read the PADRE MeDDEA flare list csv file into a TimeSeries.

    The time index is the flare start time (the "Date" column combined with the
    "Start Time (UTC)" column). Rows without a valid date or start time are skipped.

    Parameters
    ----------
    filename : Path
        Path to the flare list csv file.

    Returns
    -------
    TimeSeries
        Indexed by flare start time with columns ``end_time`` (Time), ``goes_class``,
        ``quality``, ``grafana_url``, ``public_url``, ``l0_urls`` (a list of urls per
        flare, empty if none are listed) and ``note``. When the input has a
        ``GOES Flux`` column, ``goes_flux`` contains floating-point values; blank
        values are masked. Older files without that column remain supported.
    """
    starts, ends, rows = [], [], []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        has_goes_flux = "GOES Flux" in (reader.fieldnames or [])
        for row in reader:
            try:
                date = datetime.strptime(row["Date"].strip(), "%Y-%m-%d")
                start = datetime.strptime(row["Start Time (UTC)"].strip(), "%H:%M:%S")
                end = datetime.strptime(row["End Time (UTC)"].strip(), "%H:%M:%S")
            except (ValueError, AttributeError):
                continue
            start_dt = date + timedelta(
                hours=start.hour, minutes=start.minute, seconds=start.second
            )
            end_dt = date + timedelta(
                hours=end.hour, minutes=end.minute, seconds=end.second
            )
            if end_dt < start_dt:  # flare ends after midnight
                end_dt += timedelta(days=1)
            starts.append(start_dt)
            ends.append(end_dt)
            rows.append(row)

    ts = TimeSeries(time=Time(starts, scale="utc"))
    ts["end_time"] = Time(ends, scale="utc")
    ts["goes_class"] = [r["GOES Class"].strip() for r in rows]
    if has_goes_flux:
        fluxes = []
        missing = []
        for start, row in zip(starts, rows):
            value = row["GOES Flux"]
            if value is None:
                raise ValueError(f"Missing GOES Flux field for flare at {start}.")
            value = value.strip()
            missing.append(not value)
            try:
                fluxes.append(float(value) if value else 0.0)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid GOES Flux {value!r} for flare at {start}."
                ) from exc
        ts["goes_flux"] = MaskedColumn(fluxes, mask=missing, dtype=float)
    ts["quality"] = [int(r["Quality Rating"] or 0) for r in rows]
    ts["grafana_url"] = [r["Grafana Dashboard (Internal Link)"] for r in rows]
    ts["public_url"] = [r["Public"] for r in rows]
    # multiple files are separated by "|" in the csv, placeholders like "No link" are dropped
    l0_urls = np.empty(len(rows), dtype=object)
    for i, r in enumerate(rows):
        entries = [u.strip() for u in r["Link to L0 Data"].split("|")]
        l0_urls[i] = [u for u in entries if u.startswith("http")]
    ts["l0_urls"] = l0_urls
    ts["note"] = [r["Note"] for r in rows]
    return ts


def find_flare_photon_files(flare_list: TimeSeries, level: str = "l0") -> TimeSeries:
    """Search for photon files covering the time range of each flare.

    Photon files are named by their start time only. Each is assumed to last
    one hour and is selected if its interval overlaps the flare interval.
    Intervals are start-inclusive and end-exclusive.

    Parameters
    ----------
    flare_list : TimeSeries
        Flare list as returned by `padre_meddea.io.read_flare_list`. The index is
        the flare start time and it must have ``end_time`` and ``l0_urls`` columns.
    level : str
        Data level to search, e.g. "l0" or "l1".

    Returns
    -------
    TimeSeries
        A copy of the input with ``found`` (file URLs sorted by time and
        separated by ``|``, or an empty string if none are found),
        ``matches_l0`` (whether the listed and found file names match, ignoring
        order), and ``num_found`` (number of found files) columns. Two empty
        lists match. The input is not modified.
    """
    if "l0_urls" not in flare_list.colnames:
        raise ValueError("flare_list must contain an l0_urls column.")
    # importing PADREClient registers it with Fido
    from sunpy.net import Fido
    from sunpy.net import attrs as a

    from padre_meddea.net import DataType

    results = []
    file_duration = 1 * u.hour
    for start, end in zip(flare_list.time, flare_list["end_time"]):
        query = a.Time(start - file_duration, end)
        result = Fido.search(
            query, a.Instrument.meddea, DataType.photon, a.Level(level)
        )
        files = {}
        for row in result["padre"] if "padre" in result.keys() else []:
            if not (row["Time"] < end and row["Time"] + file_duration > start):
                continue
            # keep only the latest version for each file start time
            key = row["Time"].isot
            if key not in files or row["Version"] > files[key]["Version"]:
                files[key] = row
        rows = sorted(files.values(), key=lambda r: r["Time"])
        results.append([r["url"] for r in rows])
    result = flare_list.copy()
    result["found"] = np.array(["|".join(urls) for urls in results], dtype=str)
    result["matches_l0"] = np.array(
        [
            {url.split("/")[-1] for url in listed if url}
            == {url.split("/")[-1] for url in urls if url}
            for listed, urls in zip(flare_list["l0_urls"], results)
        ],
        dtype=bool,
    )
    result["num_found"] = np.array([len(urls) for urls in results], dtype=int)
    return result


def _url_exists(url: str, timeout: float) -> bool:
    request = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, ValueError):
        return False


def check_flare_list_urls(
    flare_list: TimeSeries, timeout: float = 10, max_workers: int = 8
) -> dict[str, bool]:
    """Check whether the L0 data urls in the flare list exist.

    Parameters
    ----------
    flare_list : TimeSeries
        Flare list as returned by `read_flare_list`, must have a ``l0_urls`` column.
    timeout : float
        Seconds to wait for each url.
    max_workers : int
        Number of urls to check at the same time.

    Returns
    -------
    dict[str, bool]
        Maps each unique url in the flare list to whether it exists. Empty or
        missing urls are ignored.
    """
    urls = sorted(
        {
            url.strip()
            for urls in flare_list["l0_urls"]
            for url in (urls if urls is not None else [])
            if isinstance(url, str) and url.strip()
        }
    )
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        exists = executor.map(lambda url: _url_exists(url, timeout), urls)
        return dict(zip(urls, exists))
