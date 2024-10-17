"""
This module provides tools to analyze and manipulate spectra data.
"""

import numpy as np
from numpy.polynomial import Polynomial

import astropy.units as u
from astropy.modeling import models

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import estimate_line_parameters, fit_lines

from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.timeseries import TimeSeries
from astropy.timeseries import aggregate_downsample
from specutils import Spectrum1D


__all__ = ["get_calib_energy_func", "elist_to_spectrum", "elist_to_lc"]


def get_calib_energy_func(spectrum: Spectrum1D, line_energies, rois, deg=1):
    """Given a spectrum with known emission lines, return a function to transform between spectral axis units to energy units.

    The method used here is to fit a Gaussian model to each region of interest.
    A linear fit is then performed between the means of each Gaussian fit and the known energies.

    Parameters
    ----------
    spectrum: Spectrum1D
        A spectrum with known emission lines.
    line_energies: u.Quantity
        The energy of each line.
    rois: ndarray
        A list of regions of interest for each line.

    Returns
    -------
    func:
        A function to convert between channel space to energy space.

    Examples
    --------
    """
    if len(rois) != len(line_energies):
        raise ValueError(
            f"Number of line energies {len(line_energies)} does not match number of rois ({len(rois)})."
        )
    # if len(line_energies) < (deg + 1):
    #    raise ValueError(f"Not enough values to perform fit with degree {deg}.")
    fit_means = []
    spectral_axis_units = spectrum.spectral_axis.unit
    for this_roi in rois:
        this_region = SpectralRegion(
            this_roi[0] * spectral_axis_units, this_roi[1] * spectral_axis_units
        )
        sub_spectrum = extract_region(spectrum, this_region)
        params = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(
            amplitude=params.amplitude, mean=params.mean, stddev=params.stddev
        )
        g_fit = fit_lines(sub_spectrum, g_init)
        fit_means.append(g_fit.mean.value)
    result = Polynomial.fit(fit_means, line_energies, deg=deg)  # fit a linear model
    result_all = Polynomial.fit(fit_means, line_energies, deg=deg, full=True)
    print(result_all)
    return result.convert()  # return the function


def elist_to_lc(event_list, int_time):
    """Convert an event list to a light curve timeseries

    Parameters
    ----------
    event_list: An event list

    Returns
    -------
    ts: TimeSeries

    Examples
    --------
    """
    ts = aggregate_downsample(event_list, time_bin_size=int_time, aggregate_func=np.sum)
    return ts


def elist_to_spectrum(event_list: Table, bins=None):
    """Convert an event list to a spectrum object.

    Parameters
    ----------
    event_list: Table
        An event list


    Returns
    -------
    spectrum: Spectrum1D
    """
    if bins is None:
        bins = np.arange(0, 2**12 - 1)
    data, bins = np.histogram(event_list["atod"], bins=bins)
    result = Spectrum1D(
        flux=u.Quantity(data, "count"),
        spectral_axis=u.Quantity(bins, "pix"),
        uncertainty=StdDevUncertainty(np.sqrt(data) * u.count),
    )
    return result
