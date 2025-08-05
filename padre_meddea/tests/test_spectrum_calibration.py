import astropy.units as u
import numpy as np
import pytest
from astropy.modeling import models
from astropy.time import Time
from numpy.polynomial import Polynomial
from specutils import Spectrum1D

import padre_meddea
from padre_meddea.io.file_tools import read_file
from padre_meddea.spectrum import calibration as cal
from padre_meddea.spectrum.spectrum import SpectrumList


def test_fit_peak_parabola():
    # parabola with peak at 0
    p = Polynomial([0, 0, -1])
    x = np.arange(-5, 5)
    this_spec = Spectrum1D(spectral_axis=x * u.keV, flux=p(x) * u.ct)
    assert np.allclose([0], [cal.fit_peak_parabola(this_spec)], rtol=1e-4)
    # parabola with peak at 2
    p = Polynomial([0, 4, -1])
    x = np.arange(-5, 5)
    this_spec = Spectrum1D(spectral_axis=x * u.keV, flux=p(x) * u.ct)
    assert u.allclose([2], [cal.fit_peak_parabola(this_spec)], rtol=1e-4)
    p = models.Gaussian1D(amplitude=1, mean=2.4, stddev=1)
    x = np.arange(0, 5, 0.1)
    this_spec = Spectrum1D(spectral_axis=x * u.keV, flux=p(x) * u.ct)
    assert u.allclose([2.4], [cal.fit_peak_parabola(this_spec)], rtol=1e-2)
    p = models.Lorentz1D(amplitude=10.0, x_0=1.33, fwhm=1.0)
    this_spec = Spectrum1D(spectral_axis=x * u.keV, flux=p(x) * u.ct)
    assert u.allclose([1.33], [cal.fit_peak_parabola(this_spec)], rtol=1e-2)


def test_fit_peaks():
    # create a spectrum with lines in random places
    line_centers = [10, 20, 40] * u.keV
    p = models.Gaussian1D(amplitude=1, mean=line_centers[0].value, stddev=1)
    p += models.Lorentz1D(amplitude=10.0, x_0=line_centers[1].value, fwhm=1.0)
    p += models.Lorentz1D(amplitude=5.0, x_0=line_centers[2].value, fwhm=1.0)
    x = np.arange(0, 100, 0.1)
    this_spec = Spectrum1D(spectral_axis=x * u.keV, flux=p(x) * u.ct)
    guess_centers = line_centers + [1, 1.2, 2.3] * u.keV
    fit_centers = cal.fit_peaks(spec=this_spec, line_centers=guess_centers, window=5)
    assert len(line_centers) == len(fit_centers)
    assert u.allclose(line_centers, fit_centers, rtol=0.01)


def test_get_ql_calibration_file():
    with pytest.raises(FileNotFoundError):
        cal.get_ql_calibration_file(Time("2023-03-01T00:00"))

    result = cal.get_ql_calibration_file(Time("2025-03-01T00:00"))
    assert result.name == "20250101_ql_spec_cal.npy"
    assert result.exists()


def test_cal_spec():
    result = cal.get_ql_calibration_file(Time("2025-03-01T00:00"))
    lin_cal_params = np.load(result)

    spec_files = list((padre_meddea._test_files_directory / "spec").glob("*.fits"))
    for this_spec_file in spec_files:
        spec_list = read_file(this_spec_file)
        cal_spec_list = cal.calibrate_linear_speclist(spec_list, lin_cal_params)
        assert cal_spec_list.calibrated
        assert isinstance(cal_spec_list, SpectrumList)
