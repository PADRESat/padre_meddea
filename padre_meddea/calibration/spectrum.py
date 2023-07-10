import re

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.table import QTable
from astropy.io import ascii
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.constants.codata2018 import h, c, e, k_B

from scipy.stats.sampling import NumericalInversePolynomial

import padre_meddea

# the following are used to calculate escape lines from the Caliste-SO detector
cd_kalpha1 = 23173.6 * u.eV  # x-ray data booklet 2009
te_kalpha1 = 27472.3 * u.eV  # x-ray data booklet 2009

__all__ = ["barium_spectrum", "flare_spectrum", "setup_phgenerator_ba", "setup_phgenerator_flare"]

# load flare spectrum file
flare_spectrum_data = QTable(ascii.read(padre_meddea._data_directory / 'SOL2002-07-23_RHESSI_flare_spectrum.csv'))
for this_col in flare_spectrum_data.colnames:
    unit_str = re.findall(r'\(.*?\)', this_col)[0][1:-1]
    flare_spectrum_data[this_col].unit = u.Unit(unit_str)


@u.quantity_input
def barium_spectrum(fwhm: u.keV):
    """Provide the spectrum model from radioactive Ba133 as observed by the 
    Caliste-SO detector.

    Returns
    -------
    model : ~astropy.modeling.models

    #TODO: Add filter at low energies, add fall off due to detector thickness
    """
    ba133_lines = QTable(ascii.read(padre_meddea._data_directory / 'ba133.csv'))
    ba133_lines['energy (eV)'].unit = u.eV

    for i, this_line in enumerate(ba133_lines):
        cd_escape_line = this_line['energy (eV)'] - cd_kalpha1
        te_escape_line = this_line['energy (eV)'] - te_kalpha1
        if i == 0:
            spec = Gaussian1D(amplitude=this_line['intensity'], mean=this_line['energy (eV)'], stddev=fwhm)
        else:
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=this_line['energy (eV)'], stddev=fwhm)
        if cd_escape_line > 0 * u.eV:
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=cd_escape_line, stddev=fwhm)
        if te_escape_line > 0 * u.eV:
            spec += Gaussian1D(amplitude=this_line['intensity'], mean=te_escape_line, stddev=fwhm)
    return spec


@custom_model
def flare_spectrum(x, factor=1):
    """Provides the average spectrum of an X1 flare in x-rays.

    The minimum energy available is 3.5 keV and the highest is 14950 keV.

    Note that the original flare data is from an X4.8 class flare and is scaled
    down by a factor of 4.8

    Parameters
    ----------
    factor : int
        The factor to scale the flare

    Returns
    -------
    model : ~astropy.modeling.models
    """
    # NOTE: it may be better to interpolate the log of the spectrum for high accuracy
    func = interpolate.interp1d(
        flare_spectrum_data['Bin mean (keV)'].value,
        flare_spectrum_data[flare_spectrum_data.colnames[-1]].value,
        bounds_error=True,
        assume_sorted=True,
    )
    return func(x) * factor / 4.8


@u.quantity_input
def setup_phgenerator_ba(fwhm: u.keV):
    """Setup the random number generate to create random photons from Ba133.

    Note that this can take a few minutes to run.

    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    ba = barium_spectrum(fwhm)

    class spec:
        def pdf(self, x):
            return ba(x * u.keV)
    dist = spec()
    urng = np.random.default_rng()
    rng = NumericalInversePolynomial(dist, random_state=urng)
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
def setup_phgenerator_flare(factor):
    """Setup the random number generate to create random photons from Ba133.

    Note that this can take a few minutes to run.

    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    fa = flare_spectrum(factor)

    class spec:
        def pdf(self, x):
            return fa(x * u.keV)
    dist = spec()
    urng = np.random.default_rng()
    rng = NumericalInversePolynomial(dist, random_state=urng)
    return rng


@u.quantity_input(wavelength=u.nm, equivalencies=u.spectral())
def ThermalBremsstrahlung(wavelength, temperature: u.K):
    """
    Provides the radiation spectrum caused by the acceleration of electrons
    in the Coulomb field of another charge (generally an ion).
    This is generally referred to as free-free emission. For the case of
    thermal bremsstrahlung, the assumption is that everything is in thermal
    equilibrium. Assumes an optically thin plasma.

    Parameters
    ----------
    wavelength : `astropy.units.Quantity`
        The wavelength at which to evaluate the emission.

    temperature : `astropy.units.Quantity`
        The temperature of the plasma.

    Returns
    -------
    emission : The unscaled emission, provided without units

    TODO: Consider changing this function to an astropy model.
    """
    frequency = wavelength.to('Hz')

    emission = np.sqrt(1 / (k_B * temperature))
    emission *= np.exp(- h * frequency / (k_B * temperature.to('K')))
    emission *= GauntFactor(wavelength, temperature)

    return emission.to_value()


@u.quantity_input(wavelength=u.nm, equivalencies=u.spectral())
def GauntFactor(wavelength, temperature: u.K):
    """
    Provides the quantum mechanical correction factor for the bremsstrahlung
    emission process in an optically thin plasma.

    Parameters
    ----------
    wavelength : `astropy.units.Quantity`
        The wavelength at which to evaluate the correction factor.

    temperature : `astropy.units.Quantity`
        The temperature of the plasma.

    Returns
    -------
    correction_factor

    References
    ----------
    `Mewe et al. Astron. and Astrophys Suppl. Ser., 62, 197-254 (1985) <https://ui.adsabs.harvard.edu/abs/1985A%26AS...62..197M/abstract>`_
    """
    return 27.83 * (temperature.to('MK').value + 0.65) ** (-1.33) + 0.15 * wavelength.to('angstrom').value ** (0.34) * temperature.to('MK').value ** (0.422)
