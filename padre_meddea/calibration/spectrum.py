import re
import pickle

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.io import ascii
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.constants.codata2018 import h, c, e, k_B

from scipy.stats.sampling import NumericalInversePolynomial

import padre_meddea

# the following are used to calculate escape lines from the Caliste-SO detector
cd_kalpha1 = 23173.6 * u.eV  # x-ray data booklet 2009
te_kalpha1 = 27472.3 * u.eV  # x-ray data booklet 2009

#  TODO: add escape lines to the ba133_lines Table

__all__ = ["barium_spectrum", "flare_spectrum", "setup_phgenerator_ba", "setup_phgenerator_flare", "get_flare_rate"]

# load flare spectrum file
flare_spectrum_data = QTable(ascii.read(padre_meddea._data_directory / 'SOL2002-07-23_RHESSI_flare_spectrum.csv'))
for this_col in flare_spectrum_data.colnames:
    unit_str = re.findall(r'\(.*?\)', this_col)[0][1:-1]
    flare_spectrum_data[this_col].unit = u.Unit(unit_str)

flare_timeseries = QTable(ascii.read(padre_meddea._data_directory / 'SOL2002-07-23_GOESXRS_lightcurve.csv'))
flare_timeseries['sec_from_start'].unit = u.s

ba133_lines = QTable(ascii.read(padre_meddea._data_directory / 'ba133.csv'))
ba133_lines['energy (eV)'].unit = u.eV


@u.quantity_input
def barium_spectrum(fwhm: u.keV):
    """Provide the spectrum model from radioactive Ba133 as observed by the 
    Caliste-SO detector.

    Returns
    -------
    model : ~astropy.modeling.models

    #TODO: Add filter at low energies, add fall off due to detector thickness
    """

    for i, this_line in enumerate(ba133_lines):
        cd_escape_line = this_line['energy (eV)'] - cd_kalpha1
        te_escape_line = this_line['energy (eV)'] - te_kalpha1
        if i == 0:
            spec = Gaussian1D(amplitude=this_line['intensity'], mean=this_line['energy (eV)'], stddev=fwhm)
        else:
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=this_line['energy (eV)'], stddev=fwhm)
        if cd_escape_line > 0 * u.eV:  # TODO: need to scale down the intensity of line based on probably of escape
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=cd_escape_line, stddev=fwhm)
        if te_escape_line > 0 * u.eV:  # TODO: need to scale down the intensity of line based on probably of escape
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=te_escape_line, stddev=fwhm)
    return spec


@custom_model
def flare_spectrum(x: u.keV):
    """Provides the average spectrum of an X4.8 flare in x-rays.

    The minimum energy available is 3.5 keV and the highest is 14950 keV.

    Parameters
    ----------
    factor : int
        The factor to scale the flare

    Returns
    -------
    model : ~astropy.modeling.models
    """
    factor=1
    # NOTE: it may be better to interpolate the log of the spectrum for high accuracy
    func = interpolate.interp1d(
        flare_spectrum_data['Bin mean (keV)'].value,
        np.log10(flare_spectrum_data[flare_spectrum_data.colnames[-1]].value * factor),
        bounds_error=True,
        assume_sorted=True,
    )
    return 10 ** func(x)


@custom_model
def get_flare_rate(x: u.s):
    """Provides a times series of an X4.8 flare in x-rays as measured by GOES 
    XRS B (1 to 8 angstrom). The flare is clipped to limit flux at times when
    the power is less than 1e-5.

    x parameter : time in seconds
    y parameter : counts/s/detector

    Returns
    -------
    model : ~astropy.modeling.models
    """
    y = flare_timeseries['xrsb']
    # remove the pre and post-flare times
    y[y < 1e-5] = 1e-30
    # normalize this lightcurve to counts/s/det, from proposal calculation
    y = y / y.max() * 22421.0
    func = interpolate.interp1d(
        flare_timeseries['sec_from_start'],
        np.log10(y),
        bounds_error=True,
        assume_sorted=True,
    )
    return 10 ** func(x)


@u.quantity_input
def setup_phgenerator_ba(fwhm: u.keV):
    """Setup the random number generate to create random photons from Ba133,
    the calibration source.

    Note that this can take a few minutes to run.

    Parameters
    ----------
    fwhm : full-width-half-maximum
        Provide the resolution of the detector system.

    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    ba = barium_spectrum(fwhm)

    class spec:
        def pdf(self, x):
            return ba(x * u.keV)
    dist = spec()
    urng = np.random.default_rng(seed=42)
    rng = NumericalInversePolynomial(dist, random_state=urng, domain=[1, 150])

    return rng


def gen_random_ba_photons(ba_rvs, num):
    """Generate random photons from Ba133 source
    
    Returns
    -------
    photons : ~np.array
    """
    ba_rvs = ba_rvs.rvs(num)
    return ba_rvs


@u.quantity_input
def setup_phgenerator_flare(factor, filename=None):
    """Setup the random number generate to create random photons from Ba133.

    Note that this can take a few minutes to run.

    factor : float
        The strength of the flare relative to X class.

    filename : Path
        If set, saves the random generator object to a file for future use.
    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    fa = flare_spectrum()

    class spec:
        def pdf(self, x):
            return fa(x * u.keV)
    dist = spec()
    urng = np.random.default_rng(seed=42)
    rng = NumericalInversePolynomial(dist, random_state=urng, domain=[5, 150])
    return rng
