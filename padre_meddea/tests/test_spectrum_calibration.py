from astropy.modeling import models
from padre_meddea.spectrum import calibration as cal
from numpy.polynomial import Polynomial
from specutils import Spectrum1D
import numpy as np
import astropy.units as u


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
