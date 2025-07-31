import urllib
from collections import OrderedDict
from datetime import timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin
from typing import List

from astropy.time import Time
from sunpy.net import attrs as a
from sunpy.net.attr import SimpleAttr
from sunpy.net.dataretriever import GenericClient, QueryResponse

from padre_meddea import log
from padre_meddea.util.util import parse_science_filename


class DataType(SimpleAttr):
    """
    Attribute for specifying the data type for the search.

    Attributes
    ----------
    value : str
        The data type value.
    """


class PADREClient(GenericClient):
    """
    Data source for searching and fetching PADRE MeDDEA Data from SDAC File Servers.
    """

    baseurl = "https://umbra.nascom.nasa.gov/"

    @classmethod
    def register_values(cls):
        adict = {
            a.Provider: [("sdac", "The Solar Data Analysis Center.")],
            a.Source: [
                ("padre", "(The Solar Polarization and Directivity X-Ray Experiment)")
            ],
            a.Instrument: [
                (
                    "meddea",
                    "Measuring Directivity to Determine Electron Anisotropy (MeDDEA)",
                ),
            ],
            DataType: [
                ("spectrum", "Spectrum data from MeDDEA."),
                ("photon", "Photon data from MeDDEA."),
                ("housekeeping", "Housekeeping data from MeDDEA."),
            ],
            a.Level: [
                ("raw", "Raw Binary CCSDS Packet data"),
                ("l0", "Raw data, converted to FITS, not in physical units."),
                ("l1", "Processed data, not in physical units."),
            ],
        }
        return adict

    def search(self, *args, **kwargs) -> QueryResponse:
        """
        Query this client for a list of results.

        Parameters
        ----------
        \\*args: `tuple`
            `sunpy.net.attrs` objects representing the query.
        \\*\\*kwargs: `dict`
             Any extra keywords to refine the search.

        Returns
        -------
        A `QueryResponse` instance containing the query result.
        """
        matchdict = self._get_match_dict(*args, **kwargs)
        # Extract matchdict parameters
        instruments = matchdict.get("Instrument")
        levels = matchdict.get("Level")
        data_types = matchdict.get("DataType")
        start_time = matchdict.get("Start Time")
        end_time = matchdict.get("End Time")

        # Get search paths with data_type
        search_paths = self._get_search_paths(
            instruments, levels, data_types, start_time, end_time
        )
        log.debug(f"Search paths: {search_paths}")

        # Search each path
        all_files = []
        for path in search_paths:
            url = urljoin(self.baseurl, path)
            log.debug(f"Searching HTTP directory: {url}")
            files = self._crawl_directory(url)
            all_files.extend(files)

        # Template Replacement for DataType
        shortname_to_datatype = {
            "A0": "photon",
            "A2": "spectrum",
            "U8": "housekeeping",
        }

        # Process and return results
        metalist = []
        for file_url in all_files:
            log.debug(f"Processing file URL: {file_url}")
            info = parse_science_filename(file_url)

            # Fix the DataType Information from the Raw file and filter Raw Files with wrong DataType
            if info.get("level") == "raw":
                for shortname, longname in shortname_to_datatype.items():
                    if shortname in file_url:
                        info["descriptor"] = longname
                if info["descriptor"] not in data_types:
                    continue  # Skip files with wrong DataType

            rowdict = OrderedDict()
            rowdict["Instrument"] = info.get("instrument", "unknown")
            rowdict["Mode"] = info.get("mode", "unknown")
            rowdict["Test"] = info.get("test", False)
            rowdict["Time"] = info.get("time", "unknown")
            rowdict["Level"] = info.get("level", "unknown")
            rowdict["Version"] = info.get("version", "unknown")
            rowdict["Descriptor"] = info.get("descriptor", "unknown")
            rowdict["url"] = file_url  # Key
            metalist.append(rowdict)

        # pprint(f"Final metalist: {metalist}")
        return QueryResponse(metalist, client=self)

    def _get_search_paths(
        self,
        instruments: List[str] = None,
        levels: List[str] = None,
        data_types: List[str] = None,
        start_time: Time = None,
        end_time: Time = None,
    ):
        """Generate HTTP paths to search based on query parameters."""
        paths = []

        # Mission Name
        mission = "padre"

        time_paths = self._generate_time_paths(start_time, end_time)
        # Combine all path components
        for instrument in instruments:
            for level in levels:
                if level == "raw":
                    for time_path in time_paths:
                        # For raw data, do not include data type in the path
                        paths.append(
                            f"{mission}/{mission}-{instrument}/{level}/{time_path}/"
                        )
                else:
                    # For other levels, include data type in the path
                    for data_type in data_types:
                        for time_path in time_paths:
                            # For other levels, include data type in the path
                            paths.append(
                                f"{mission}/{mission}-{instrument}/{level}/{data_type}/{time_path}/"
                            )
        return paths

    def _generate_time_paths(self, start_time: Time, end_time: Time):
        """
        Generate all year/month/day path components between start_time and end_time.

        Parameters
        ----------
        start_time : astropy.time.Time
            Start time in ISO format (e.g., '2025-05-04')
        end_time : astropy.time.Time
            End time in ISO format (e.g., '2025-07-07')

        Returns
        -------
        list
            List of path strings in format 'YYYY/MM/DD'
        """
        # Parse the ISO format times
        start_date = start_time.datetime
        end_date = end_time.datetime

        # Initialize empty list for paths
        time_paths = []

        # Iterate through each day in the range
        current_date = start_date
        while current_date <= end_date:
            # Format as YYYY/MM/DD
            path = (
                f"{current_date.year}/{current_date.month:02d}/{current_date.day:02d}"
            )
            time_paths.append(path)

            # Move to next day
            current_date += timedelta(days=1)

        log.debug(
            f"Generated {len(time_paths)} time paths from {start_time} to {end_time}"
        )
        return time_paths

    def _crawl_directory(self, url):
        """Directory crawler using only standard library."""

        class LinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.links = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for attr, value in attrs:
                        if attr == "href":
                            self.links.append(value)

        files = []
        try:
            with urllib.request.urlopen(url) as response:
                html = response.read().decode("utf-8")

            parser = LinkParser()
            parser.feed(html)

            for href in parser.links:
                # Skip parent directory links and query parameters
                if not href or href.startswith("?") or href == "../":
                    continue

                full_url = urljoin(url, href)

                # Don't crawl up: make sure we're still below our starting point
                if not full_url.startswith(self.baseurl) or len(full_url) < len(
                    self.baseurl
                ):
                    continue

                elif href.lower().endswith(".fits") or href.lower().endswith(".dat"):
                    files.append(full_url)

            return files
        except Exception as e:
            log.debug(f"Error processing {url}: {e}")
            return []
