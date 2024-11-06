import pytest

import astropy.units as u
from specutils import Spectrum1D

import padre_meddea
from padre_meddea.io.amptek import read_mca

mca_file = padre_meddea._test_files_directory / "minix_20kV_15uA_sdd.mca"

spec = read_mca(mca_file)


def test_read_mca():
    """Test that we can open an mca file and get a spectrum object."""
    assert isinstance(spec, Spectrum1D)


def test_read_mca_rate():
    """Test that count_rate option works"""
    spec = read_mca(mca_file, count_rate=True)
    assert spec.flux.unit == u.ct / u.s


@pytest.mark.parametrize(
    "keyword", ["filename", "livetime", "rate", "dtimfrac"]
)
def test_read_mca_meta_added(keyword):
    """Test that a few added meta data items."""
    assert keyword in spec.meta


@pytest.mark.parametrize(
    "keyword", ["LIVETIME", "SERIALNU", "REALTIME", "STARTTIM"]
)
def test_read_mca_meta_orig(keyword):
    """Test that a few original meta data items are present."""
    assert keyword in spec.meta["header"]


def test_read_mca_calib():
    """Test that reading the calibration data worked and generated a valid function."""
    f = spec.meta["calib"]
    calib_energy_axis = f(spec.spectral_axis.value)
    assert calib_energy_axis.min() > -1
    assert calib_energy_axis.max() < 23


def test_read_mca_rois():
    """Test that the ROIs have been read in from the file."""
    rois = spec.meta["roi"]
    assert len(rois) == 5
    assert rois[0].lower == 150.0 * u.pix
    assert rois[0].upper == 250.0 * u.pix
    assert rois[3].lower == 1040.0 * u.pix
    assert rois[3].upper == 1120.0 * u.pix
