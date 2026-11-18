"""
Provides a reader for the PADRE MeDDEA flare list.
"""

import csv
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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
        ``quality``, ``grafana_url``, ``public_url``, ``l0_urls`` (a list of urls per flare, empty if none are listed) and ``note``.
    """
    starts, ends, rows = [], [], []
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
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
