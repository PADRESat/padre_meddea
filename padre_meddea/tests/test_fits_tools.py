import os
import platform
from pathlib import Path
import tempfile

import numpy as np
import pytest
import astropy.io.fits as fits
from astropy.table import Table
from astropy.time import Time

from padre_meddea.io.fits_tools import (
    _init_hdul_structure,
    concatenate_files,
    get_hdu_data_times,
)
from padre_meddea.util.util import calc_time

TEST_DATA_DIR = Path(__file__).parent.parent / "data/test"


# Fix for memmap issue and files not closing on Windows
@pytest.fixture(autouse=True)
def patch_fits_open_for_windows(monkeypatch):
    """Ensure fits.open is called with memmap=False on Windows."""
    if platform.system().lower() == "windows":
        original_open = fits.open

        def patched_open(*args, **kwargs):
            kwargs.setdefault("memmap", False)
            return original_open(*args, **kwargs)

        monkeypatch.setattr(fits, "open", patched_open)


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield Path(tmpdirname)


@pytest.fixture
def photon_sci_hdu_dict():
    """Create a dictionary representing a photon SCI HDU with test data."""
    # Create test data with known timestamps
    pkttimes = np.array([100, 200, 300], dtype=np.int64)
    pktclock = np.array([1, 2, 3], dtype=np.int32)
    clocks = np.array([10, 20, 30], dtype=np.int64)

    # Create table with the test data
    data = Table({"pkttimes": pkttimes, "pktclock": pktclock, "clocks": clocks})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "SCI"
    header["BTYPE"] = "photon"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "SCI"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes, pktclock, clocks)

    return hdu_dict, expected_times


@pytest.fixture
def photon_pkt_hdu_dict():
    """Create a dictionary representing a photon PKT HDU with test data."""
    # Create test data with known timestamps
    pkttimes = np.array([400, 500, 600], dtype=np.int64)
    pktclock = np.array([4, 5, 6], dtype=np.int32)

    # Create table with the test data
    data = Table({"pkttimes": pkttimes, "pktclock": pktclock})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "PKT"
    header["BTYPE"] = "photon"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "PKT"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes, pktclock)

    return hdu_dict, expected_times


@pytest.fixture
def hk_pkttimes_hdu_dict():
    """Create a dictionary representing a housekeeping HK HDU with pkttimes."""
    # Create test data with known timestamps
    pkttimes = np.array([700, 800, 900], dtype=np.int64)

    # Create table with the test data
    data = Table({"pkttimes": pkttimes})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "HK"
    header["BTYPE"] = "housekeeping"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "HK"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes)

    return hdu_dict, expected_times


@pytest.fixture
def hk_timestamp_hdu_dict():
    """Create a dictionary representing a housekeeping HK HDU with timestamp."""
    # Create test data with known timestamps
    timestamp = np.array([1000, 1100, 1200], dtype=np.int64)

    # Create table with the test data
    data = Table({"timestamp": timestamp})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "HK"
    header["BTYPE"] = "housekeeping"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "HK"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(timestamp)

    return hdu_dict, expected_times


@pytest.fixture
def hk_read_hdu_dict():
    """Create a dictionary representing a housekeeping READ HDU."""
    # Create test data with known timestamps
    pkttimes = np.array([1300, 1400, 1500], dtype=np.int64)
    pktclock = np.array([7, 8, 9], dtype=np.int32)

    # Create table with the test data
    data = Table({"pkttimes": pkttimes, "pktclock": pktclock})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "READ"
    header["BTYPE"] = "housekeeping"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "READ"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes, pktclock)

    return hdu_dict, expected_times


@pytest.fixture
def spectrum_pkt_hdu_dict():
    """Create a dictionary representing a spectrum PKT HDU."""
    # Create test data with known timestamps
    pkttimes = np.array([1600, 1700, 1800], dtype=np.int64)
    pktclock = np.array([10, 11, 12], dtype=np.int32)

    # Create table with the test data
    data = Table({"pkttimes": pkttimes, "pktclock": pktclock})

    # Create header with the required metadata
    header = fits.Header()
    header["EXTNAME"] = "PKT"
    header["BTYPE"] = "spectrum"

    # Create the HDU dictionary structure
    hdu_dict = {1: {"header": header, "data": data, "type": "bintable", "name": "PKT"}}

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes, pktclock)

    return hdu_dict, expected_times


@pytest.fixture
def spectrum_spec_hdu_dict():
    """Create a dictionary representing a spectrum SPEC HDU with corresponding PKT HDU."""
    # Create test data for PKT HDU (needed for SPEC timing)
    pkttimes = np.array([1900, 2000, 2100], dtype=np.int64)
    pktclock = np.array([13, 14, 15], dtype=np.int32)

    pkt_data = Table({"pkttimes": pkttimes, "pktclock": pktclock})

    pkt_header = fits.Header()
    pkt_header["EXTNAME"] = "PKT"
    pkt_header["BTYPE"] = "spectrum"

    # Create test data for SPEC HDU
    spec_data = Table(
        {
            "channel": np.array([0, 1, 2], dtype=np.int32),
            "counts": np.array([10, 20, 30], dtype=np.int32),
        }
    )

    spec_header = fits.Header()
    spec_header["EXTNAME"] = "SPEC"
    spec_header["BTYPE"] = "spectrum"

    # Create the HDU dictionary structure with both PKT and SPEC HDUs
    hdu_dict = {
        1: {"header": pkt_header, "data": pkt_data, "type": "bintable", "name": "PKT"},
        2: {
            "header": spec_header,
            "data": spec_data,
            "type": "bintable",
            "name": "SPEC",
        },
    }

    # Calculate expected times using calc_time for verification
    expected_times = calc_time(pkttimes, pktclock)

    return hdu_dict, expected_times


@pytest.fixture
def empty_data_hdu_dict():
    """Create an HDU dictionary with empty data tables for testing."""
    # Create empty tables for different HDUs
    empty_sci_data = Table(
        {
            "pkttimes": np.array([], dtype=np.int64),
            "pktclock": np.array([], dtype=np.int32),
            "clocks": np.array([], dtype=np.int64),
        }
    )

    sci_header = fits.Header()
    sci_header["EXTNAME"] = "SCI"
    sci_header["BTYPE"] = "photon"

    hdu_dict = {
        1: {
            "header": sci_header,
            "data": empty_sci_data,
            "type": "bintable",
            "name": "SCI",
        }
    }

    # For empty data, expect an empty Time array
    expected_times = Time([], format="iso")

    return hdu_dict, expected_times


def test_photon_sci_hdu_times(photon_sci_hdu_dict):
    """Test extracting times from photon SCI HDU."""
    hdu_dict, expected_times = photon_sci_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "SCI")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_photon_pkt_hdu_times(photon_pkt_hdu_dict):
    """Test extracting times from photon PKT HDU."""
    hdu_dict, expected_times = photon_pkt_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "PKT")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_hk_pkttimes_hdu_times(hk_pkttimes_hdu_dict):
    """Test extracting times from housekeeping HK HDU with pkttimes."""
    hdu_dict, expected_times = hk_pkttimes_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "HK")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_hk_timestamp_hdu_times(hk_timestamp_hdu_dict):
    """Test extracting times from housekeeping HK HDU with timestamp."""
    hdu_dict, expected_times = hk_timestamp_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "HK")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_hk_read_hdu_times(hk_read_hdu_dict):
    """Test extracting times from housekeeping READ HDU."""
    hdu_dict, expected_times = hk_read_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "READ")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_spectrum_pkt_hdu_times(spectrum_pkt_hdu_dict):
    """Test extracting times from spectrum PKT HDU."""
    hdu_dict, expected_times = spectrum_pkt_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "PKT")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_spectrum_spec_hdu_times(spectrum_spec_hdu_dict):
    """Test extracting times from spectrum SPEC HDU."""
    hdu_dict, expected_times = spectrum_spec_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "SPEC")

    assert len(result_times) == len(expected_times)
    assert np.all(np.isclose(result_times.mjd, expected_times.mjd))


def test_empty_data_times(empty_data_hdu_dict):
    """Test handling of empty data tables."""
    hdu_dict, expected_times = empty_data_hdu_dict
    result_times = get_hdu_data_times(hdu_dict, "SCI")

    assert len(result_times) == 0


def test_missing_hdu():
    """Test error when HDU name is not found."""
    hdu_dict = {
        0: {"header": fits.Header(), "data": None, "type": "primary", "name": "PRIMARY"}
    }

    with pytest.raises(
        ValueError, match="No HDU with name MISSING found in HDU dictionary"
    ):
        get_hdu_data_times(hdu_dict, "MISSING")


def test_unsupported_data_type():
    """Test error when data type is not supported."""
    unsupported_hdu_dict = {
        1: {
            "header": fits.Header([("BTYPE", "unsupported")]),
            "data": Table({"data": [1, 2, 3]}),
            "type": "bintable",
            "name": "TEST",
        }
    }

    with pytest.raises(
        ValueError,
        match="File contents of TEST not recognized for data type unsupported",
    ):
        get_hdu_data_times(unsupported_hdu_dict, "TEST")


def test_missing_pkt_hdu_for_spec():
    """Test error when PKT HDU is missing for SPEC HDU."""
    # Create a spectrum HDUL dict without PKT HDU
    spec_only_hdu_dict = {
        2: {
            "header": fits.Header([("BTYPE", "spectrum")]),
            "data": Table({"channel": [1, 2, 3]}),
            "type": "bintable",
            "name": "SPEC",
        }
    }

    with pytest.raises(ValueError, match="No PKT HDU found for SPEC HDU"):
        get_hdu_data_times(spec_only_hdu_dict, "SPEC")


def validate_hdul_members(hdul: fits.HDUList):
    data_type = hdul[0].header.get("BTYPE", "").lower()

    # Check for PROVENANCE HDU in all files
    hdu_names = [hdu.name for hdu in hdul]
    assert "PROVENANCE" in hdu_names, f"File missing PROVENANCE HDU. Found: {hdu_names}"

    # Switch based on data type to validate the expected HDU structure
    if data_type == "photon":
        # Check if photon file has the expected 4 HDUs (PRIMARY, SCI, PKT, PROVENANCE)
        assert len(hdul) == 4, f"Photon file should have 4 HDUs, found {len(hdul)}"

        # Verify expected HDU names
        assert "SCI" in hdu_names, f"Photon file missing SCI HDU. Found: {hdu_names}"
        assert "PKT" in hdu_names, f"Photon file missing PKT HDU. Found: {hdu_names}"

    elif data_type == "spectrum":
        # Check if spectrum file has the expected 4 HDUs (PRIMARY, SPEC, PKT, PROVENANCE)
        assert len(hdul) == 4, f"Spectrum file should have 4 HDUs, found {len(hdul)}"

        # Verify expected HDU names
        assert "SPEC" in hdu_names, (
            f"Spectrum file missing SPEC HDU. Found: {hdu_names}"
        )
        assert "PKT" in hdu_names, f"Spectrum file missing PKT HDU. Found: {hdu_names}"

    elif data_type == "housekeeping":
        # Check if housekeeping file has the expected 4 HDUs (PRIMARY, HK, READ, PROVENANCE)
        assert len(hdul) == 4, (
            f"Housekeeping file should have 4 HDUs, found {len(hdul)}"
        )

        # Verify expected HDU names
        assert "HK" in hdu_names, (
            f"Housekeeping file missing HK HDU. Found: {hdu_names}"
        )
        assert "READ" in hdu_names, (
            f"Housekeeping file missing READ HDU. Found: {hdu_names}"
        )

    else:
        assert False, f"Unknown data type: {data_type}"


@pytest.mark.parametrize(
    "input_files, expected_outputs, expected_parentxt, additional_file, expected_additional_outputs, additional_parentext, expected_provenance_rows",
    [
        # Corrected eventlist (single-day) case
        (
            [
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v1.0.0.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
            [
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v1.0.1.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:53:11.353",
                    "date-end": "2025-05-04T05:53:11.830",
                    "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T07:37:49.472",
                    "date-end": "2025-05-04T07:37:49.476",
                    "filename": "padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:03:30.385",
                    "date-end": "2025-05-04T08:03:30.390",
                    "filename": "padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
                },
            ],
        ),
        # Corrected eventlist (Multi-day) case
        (
            [
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            ],
            ["padre_meddea_l1_photon_20250504T000000_v1.0.0.fits"],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
            [
                TEST_DATA_DIR
                / "eventlist/padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_photon_20250504T000000_v1.0.1.fits",
                "padre_meddea_l1_photon_20250505T000000_v1.0.0.fits",
            ],
            "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits, padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits, padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits, padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:53:11.353",
                    "date-end": "2025-05-04T05:53:11.830",
                    "filename": "padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T07:37:49.472",
                    "date-end": "2025-05-04T07:37:49.476",
                    "filename": "padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:03:30.385",
                    "date-end": "2025-05-04T08:03:30.390",
                    "filename": "padre_meddea_l0test_photon_20250504T080330_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:32:34.299",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits",
                },
            ],
        ),
        # Corrected HK case
        (
            [
                TEST_DATA_DIR
                / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
                TEST_DATA_DIR
                / "hk/padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
                TEST_DATA_DIR
                / "hk/padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v1.0.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
            [
                TEST_DATA_DIR
                / "hk/padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_housekeeping_20250504T000000_v1.0.1.fits",
                "padre_meddea_l1_housekeeping_20250505T000000_v1.0.0.fits",
            ],
            "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits, padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T05:51:38.000",
                    "date-end": "2025-05-04T05:53:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:53:38.000",
                    "date-end": "2025-05-04T05:55:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:55:38.000",
                    "date-end": "2025-05-04T05:57:08.000",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055508_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T05:57:38.000",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_housekeeping_20250504T055708_v0.1.0.fits",
                },
            ],
        ),
        # Corrected spec case
        (
            [
                TEST_DATA_DIR
                / "spec/padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits",
                TEST_DATA_DIR
                / "spec/padre_meddea_l0test_spectrum_20250504T081521_v0.1.0.fits",
                TEST_DATA_DIR
                / "spec/padre_meddea_l0test_spectrum_20250504T103811_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v1.0.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T081521_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T103811_v0.1.0.fits",
            [
                TEST_DATA_DIR
                / "spec/padre_meddea_l0test_spectrum_20250504T114921_v0.1.0.fits",
            ],
            [
                "padre_meddea_l1_spectrum_20250504T000000_v1.0.1.fits",
                "padre_meddea_l1_spectrum_20250505T000000_v1.0.0.fits",
            ],
            "padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T081521_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T103811_v0.1.0.fits, padre_meddea_l0test_spectrum_20250504T114921_v0.1.0.fits",
            [
                {
                    "date-beg": "2025-05-04T07:04:11.349",
                    "date-end": "2025-05-04T07:05:41.349",
                    "filename": "padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T08:15:21.363",
                    "date-end": "2025-05-04T08:16:51.364",
                    "filename": "padre_meddea_l0test_spectrum_20250504T081521_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T10:38:11.392",
                    "date-end": "2025-05-04T10:39:51.392",
                    "filename": "padre_meddea_l0test_spectrum_20250504T103811_v0.1.0.fits",
                },
                {
                    "date-beg": "2025-05-04T11:49:21.406",
                    "date-end": "2025-05-04T23:59:59.999",
                    "filename": "padre_meddea_l0test_spectrum_20250504T114921_v0.1.0.fits",
                },
            ],
        ),
    ],
)
def test_concatenate_fits_cases(
    tmpdir,
    input_files,
    expected_outputs,
    expected_parentxt,
    additional_file,
    expected_additional_outputs,
    additional_parentext,
    expected_provenance_rows,
):
    os.chdir(tmpdir)

    # Initial concatenation
    output_files = concatenate_files(input_files)
    assert all(output_file.exists() for output_file in output_files)
    assert len(output_files) == len(expected_outputs)
    for output_file, expected_name in zip(output_files, expected_outputs):
        assert output_file.name == expected_name

    output_file = output_files[0]
    expected_parentxt_list = expected_parentxt.split(", ")

    # Check primary header and provenance
    with fits.open(output_file, memmap=False) as hdul:
        header = hdul[0].header
        assert "PARENTXT" in header
        assert header["PARENTXT"] == expected_parentxt

        provenance_hdu = next((hdu for hdu in hdul if hdu.name == "PROVENANCE"), None)
        assert provenance_hdu is not None, "PROVENANCE HDU not found"

        provenance_table = Table(provenance_hdu.data)
        assert len(provenance_table) == len(expected_parentxt_list)
        assert {"FILENAME", "DATE_BEG", "DATE_END"} <= set(provenance_table.colnames)

        # Handle both dict or list of dicts
        assert (
            provenance_table[0]["FILENAME"] == expected_provenance_rows[0]["filename"]
        )
        assert (
            str(provenance_table[0]["DATE_BEG"])
            == expected_provenance_rows[0]["date-beg"]
        )
        assert (
            str(provenance_table[0]["DATE_END"])
            == expected_provenance_rows[0]["date-end"]
        )

    # Add additional file and re-validate
    output_files = concatenate_files(additional_file, existing_file=output_file)
    assert all(output_file.exists() for output_file in output_files)
    assert len(output_files) == len(expected_additional_outputs)
    for output_file, expected_name in zip(output_files, expected_additional_outputs):
        assert output_file.name == expected_name

    output_file = output_files[0]

    with fits.open(output_file, memmap=False) as hdul:
        validate_hdul_members(hdul)

        header = hdul[0].header
        assert "PARENTXT" in header
        assert header["PARENTXT"] == additional_parentext

        provenance_hdu = next((hdu for hdu in hdul if hdu.name == "PROVENANCE"), None)
        assert provenance_hdu is not None, "PROVENANCE HDU not found after append"

        provenance_table = Table(provenance_hdu.data)
        additional_parentxt_list = additional_parentext.split(", ")
        assert len(provenance_table) == len(additional_parentxt_list)

        expected_rows = []

        for i, filename in enumerate(additional_parentxt_list):
            assert provenance_table["FILENAME"][i] == filename
            assert provenance_table["DATE_BEG"][i] is not None
            assert provenance_table["DATE_END"][i] is not None

            expected_row = {
                "filename": provenance_table["FILENAME"][i],
                "date-beg": provenance_table["DATE_BEG"][i],
                "date-end": provenance_table["DATE_END"][i],
            }

            expected_rows.append(expected_row)

        assert expected_rows == expected_provenance_rows, (
            "Expected rows do not match actual rows in provenance table"
        )

    if len(output_files) > 1:
        secondary_output_file = output_files[1]

        with fits.open(secondary_output_file, memmap=False) as hdul:
            validate_hdul_members(hdul)

            header = hdul[0].header
            assert "PARENTXT" in header

            # Get last item in PARENTXT since that is the multi-date test case
            additional_parentext = additional_parentext.split(", ")[-1]

            assert header["PARENTXT"] == additional_parentext

            provenance_hdu = next(
                (hdu for hdu in hdul if hdu.name == "PROVENANCE"), None
            )
            assert provenance_hdu is not None, "PROVENANCE HDU not found after append"

            provenance_table = Table(provenance_hdu.data)
            additional_parentxt_list = additional_parentext.split(", ")
            assert len(provenance_table) == 1
            assert {"FILENAME", "DATE_BEG", "DATE_END"} <= set(
                provenance_table.colnames
            )
            assert provenance_table["FILENAME"][0] == additional_parentext
            assert provenance_table["DATE_BEG"][0] == "2025-05-05T00:00:00.000"

    # Test if you try to concatenate a file that is already concatenated, it logs a warning with logging.warning and does not change the file
    empty_list = concatenate_files(
        files_to_combine=additional_file, existing_file=output_file
    )
    assert empty_list == [], "Expected empty list when file is already concatenated"


@pytest.mark.parametrize(
    "input_files",
    [
        [
            TEST_DATA_DIR
            / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
            TEST_DATA_DIR
            / "eventlist/padre_meddea_l0test_photon_20250504T073749_v0.1.0.fits",
        ],
        [
            TEST_DATA_DIR
            / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
            TEST_DATA_DIR
            / "hk/padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits",
        ],
        [
            TEST_DATA_DIR
            / "spec/padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits",
            TEST_DATA_DIR
            / "spec/padre_meddea_l0test_spectrum_20250504T081521_v0.1.0.fits",
        ],
    ],
)
def test_eventlist_concatenate(input_files):
    """Test that the data is in the right order."""

    output_files = concatenate_files(files_to_combine=input_files)
    outfile = _init_hdul_structure(output_files[0])

    # check all hdu but ignore the first hdu which does not contain data
    for outhdu_idx in range(1, len(outfile)):
        # check that the data is the same
        times = get_hdu_data_times(outfile, outfile[outhdu_idx]["name"])
        assert np.all(sorted(times) == times)


# add test to check that no data was lost for each file
# add test to check that data is in the right order
