import os
import pytest
import tempfile
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler

from astropy.time import Time
from sunpy.net import Fido
from sunpy.net import attrs as a

from padre_meddea.net.client import PADREClient, DataType


@pytest.fixture
def http_file_server():
    """
    Helper: Temporary HTTP server serving a directory
    """
    # Set Config to use the padre mission
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files and directories
        base_path = "padre/padre-meddea/"

        # Level 0 and Level 1 Data Files
        date_path = "2025/05/04"
        filename_date = "20250504T000000"
        levels_to_create = ["l0", "l1"]
        data_types_to_create = ["photon", "spectrum", "housekeeping"]
        for level in levels_to_create:
            for data_type in data_types_to_create:
                # Create the URL Path / Folder to Test Data
                temp_folder = os.path.join(
                    tmpdir, base_path, level, data_type, date_path
                )
                os.makedirs(temp_folder, exist_ok=True)
                # Create Test Files
                test_file = os.path.join(
                    temp_folder,
                    f"padre_meddea_{level}_{data_type}_{filename_date}_v0.1.0.fits",
                )
                with open(test_file, "w") as f:
                    f.write(f"FITS Data for {level} {data_type} {filename_date}")

        # Create a directory for raw data
        level = "raw"
        raw_data_types = ["A0", "A2", "U8"]  # Photon, Spectrum, Housekeeping
        raw_filename_date = "250504010101"
        raw_folder = os.path.join(tmpdir, base_path, level, date_path)
        os.makedirs(raw_folder, exist_ok=True)
        for raw_type in raw_data_types:
            raw_file = os.path.join(
                raw_folder,
                f"PADREMD{raw_type}_{raw_filename_date}.DAT",
            )
            with open(raw_file, "w") as f:
                f.write(f"Raw Data for {raw_type} {raw_filename_date}")

        # Start HTTP server in a thread
        class QuietHandler(SimpleHTTPRequestHandler):
            def log_message(self, format, *args):
                pass

        server = HTTPServer(("localhost", 0), QuietHandler)
        port = server.server_port
        thread = threading.Thread(target=server.serve_forever)
        cwd = os.getcwd()
        os.chdir(tmpdir)
        thread.start()
        try:
            yield f"http://localhost:{port}/"
        finally:
            server.shutdown()
            thread.join()
            os.chdir(cwd)


@pytest.mark.parametrize(
    "instruments,levels,data_types,expected_paths",
    [
        (
            ["meddea"],
            ["l1"],
            ["housekeeping"],
            ["padre/padre-meddea/l1/housekeeping/2025/05/01/"],
        ),
        (
            ["meddea"],
            ["l1"],
            ["housekeeping", "spectrum", "photon"],
            [
                "padre/padre-meddea/l1/housekeeping/2025/05/01/",
                "padre/padre-meddea/l1/spectrum/2025/05/01/",
                "padre/padre-meddea/l1/photon/2025/05/01/",
            ],
        ),
        (
            ["meddea"],
            ["raw", "l0", "l1"],
            ["housekeeping"],
            [
                "padre/padre-meddea/raw/2025/05/01/",
                "padre/padre-meddea/l0/housekeeping/2025/05/01/",
                "padre/padre-meddea/l1/housekeeping/2025/05/01/",
            ],
        ),
    ],
)
def test_search_paths(
    http_file_server, instruments, levels, data_types, expected_paths
):
    client = PADREClient()
    paths = client._get_search_paths(
        instruments=instruments,
        levels=levels,
        data_types=data_types,
        start_time=Time("2025-05-01"),
        end_time=Time("2025-05-01"),
    )
    assert sorted(paths) == sorted(expected_paths)


def test_fido_search(http_file_server, monkeypatch):
    """
    Test getting spectrum data from the PADREClient.
    """
    # Patch the baseurl with our test server URL
    monkeypatch.setattr("padre_meddea.net.client.PADREClient.baseurl", http_file_server)

    # Test Getting Raw, Level 0, and Level 1 Spectrum Data All Together
    result = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & DataType.spectrum
    )
    padre_results = result["padre"]
    assert len(padre_results) == 3
    assert all(
        [
            "raw" in padre_results["Level"],
            "l0" in padre_results["Level"],
            "l1" in padre_results["Level"],
        ]
    )

    # Test Getting All Level 1 Data Across All Data Types
    result = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & a.Level.l1
    )
    padre_results = result["padre"]
    assert len(padre_results) == 3
    assert all(
        [
            "housekeeping" in padre_results["Descriptor"],
            "spectrum" in padre_results["Descriptor"],
            "photon" in padre_results["Descriptor"],
        ]
    )
    # Assert all Results are Level 1
    assert all(["l1" == level for level in padre_results["Level"]])

    # Test Getting L1 Photon Data
    result = Fido.search(
        a.Time("2025-05-01", "2025-05-05")
        & a.Instrument.meddea
        & a.Level.l1
        & DataType.photon
    )
    padre_results = result["padre"]
    assert len(padre_results) == 1
    assert all(
        ["photon" == padre_results["Descriptor"][0], "l1" == padre_results["Level"][0]]
    )

    # Test Getting all RAW Data
    result = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & a.Level.raw
    )
    padre_results = result["padre"]
    assert len(padre_results) == 3
    assert all(
        [
            "housekeeping" in padre_results["Descriptor"],
            "spectrum" in padre_results["Descriptor"],
            "photon" in padre_results["Descriptor"],
        ]
    )
    # Assert all Results are RAW
    assert all(["raw" == level for level in padre_results["Level"]])
