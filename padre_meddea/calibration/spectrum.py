"""
This module provides tools to analyze and manipulate spectra data.
"""

import numpy as np
from numpy.polynomial import Polynomial

import astropy.units as u
from astropy.modeling import models
from astropy.time import Time
from astropy.nddata import StdDevUncertainty
from astropy.table import Table
from astropy.timeseries import TimeSeries, aggregate_downsample

from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import estimate_line_parameters, fit_lines

from padre_meddea.util.util import pixelid_to_str, parse_pixelids, channel_to_pixel
import padre_meddea

__all__ = [
    "get_calib_energy_func",
    "elist_to_spectrum",
    "elist_to_lc",
    "PhotonList",
    "SpectrumList",
]


class PhotonList:
    """Photon data container for MeDDEA photon measurements

    Parameters
    ----------
    pkt_list : TimeSeries
        The time series of photon packet header data.
    event_list : TimeSeries
        The time series of event data
    """

    def __init__(self, pkt_list: TimeSeries, event_list: TimeSeries):
        self.pkt_list = pkt_list
        self.event_list = event_list

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.event_list[key]
        elif isinstance(key, slice):
            if isinstance(key.start, Time) and isinstance(key.stop, Time):
                pkt_ind = (self.pkt_list.time > key.start) * (
                    self.pkt_list.time < key.stop
                )
                ph_ind = (self.event_list.time > key.start) * (
                    self.event_list.time < key.stop
                )
                return type(self)(self.pkt_list[pkt_ind], self.event_list[ph_ind])
        return self

    def spectrum(
        self, asic_num: int, pixel_num: int, bins=None, baseline_sub: bool = False
    ) -> Spectrum1D:
        """
        Create a spectrum

        Parameters
        ----------
        asic_num : int
            The asic or detector number (0 to 3)
        pixel_num : int
            The pixel number (0 to 11)
        bins : np.array
            The bin edges for the spectrum (see ~np.histogram).
            If None, then uses np.arange(0, 2**12 - 1)
        baseline_sub : bool
            If True, then baseline measurements are subtracted if they exist
            Note: not yet implemented.

        Returns
        -------
        spectrum : Spectrum1D
        """
        if bins is None:
            bins = np.arange(0, 2**12 - 1)
        this_event_list = self._slice_event_list(asic_num, pixel_num)
        data, new_bins = np.histogram(this_event_list["atod"], bins=bins)
        # the spectral axis is at the center of the bins
        result = Spectrum1D(
            flux=u.Quantity(data, "count"),
            spectral_axis=u.Quantity(bins, "pix"),
            uncertainty=StdDevUncertainty(np.sqrt(data) * u.count),
        )
        return result

    def lightcurve(
        self, asic_num: int, pixel_num: int, int_time: u.Quantity[u.s], step: int = 10
    ) -> TimeSeries:
        """
        Create a light curve

        Parameters
        ----------
        asic_num : int
            The asic or detector number (0 to 3)
        pixel_num : int
            The pixel number (0 to 11)
        int_time : u.Quantity[u.s]
            The integration time
        step : int
            To speed up processing, skip every `step` photons.
            Default is ten.
            The light curve count rate is corrected by multiplying by `step`.

        Returns
        -------
        lc : TimeSeries
        """
        if (asic_num is None) and (pixel_num is None):
            this_event_list = self.event_list
        else:
            this_event_list = self.slice_event_list(asic_num, pixel_num)
        this_event_list = TimeSeries(
            time=self.event_list.time[::step]
        )  # not sure why this is necessary
        this_event_list["count"] = np.ones(len(this_event_list))
        ts = aggregate_downsample(
            this_event_list, time_bin_size=int_time, aggregate_func=np.sum
        )
        ts["count"] *= step
        return ts

    def _slice_event_list(self, asic_num: int, pixel_num: int) -> TimeSeries:
        """Slice the event list to only contain events from asic_num and pixel_num"""
        ind = (self.event_list["pixel"] == pixel_num) * (
            self.event_list["asic"] == asic_num
        )
        return self.event_list[ind]


class SpectrumList:
    """
    A spectrum data container to store MeDDEA spectrum data

    Parameters
    ----------
    pkt_spec : TimeSeries
        The time series of spectrum packet header data.
    specs : Spectrum1D
        The spectrum cube
    pixel_ids : np.array
        The pixel id array

    Raises
    ------
    ValueError
        If pixel arrays are found to change.

    Examples
    --------
    >>> from padre_meddea.calibration.spectrum import SpectrumList
    >>> from padre_meddea.io import read_raw_file
    >>> from astropy.time import Time
    >>> tr = [Time('2025-01-30 10:52'), Time('2025-01-30 11:05')]
    >>> spec_list = read_raw_file(Path("data/padre_meddea_l0test_calba_20250130T104700_v0.3.0.bin"))
    >>> this_spec_list = spec_list[tr[0]:tr[1]]
    >>> this_spectrum = this_spec_list.spectrum(asic_num=0, pixel_num=0)
    """

    def __init__(self, pkt_spec: TimeSeries, specs, pixel_ids):
        self.bins = padre_meddea.HIST_BINS
        self.time = pkt_spec.time
        self.pkt_spec = pkt_spec
        self.specs = specs
        self._pixel_ids = pixel_ids
        if np.all(np.unique(pixel_ids) == sorted(pixel_ids[0])):
            self.pixel_ids = pixel_ids[0]
            self.pixel_str = pixelid_to_str(pixel_ids[0])
            self.asics, self.channel_nums = parse_pixelids(pixel_ids[0])
            self.pixel_nums = channel_to_pixel(self.channel_nums)
        else:
            raise ValueError("Found change in pixel ids")
        self.index = len(pkt_spec)

    def spectrum(self, asic_num: int, pixel_num: int):
        """Create a spectrum, integrates over all times

        Parameters
        ----------
        asic_num : int
            The asic or detector number (0 to 3)
        pixel_num : int
            The pixel number (0 to 11)

        Raises
        ------
        ValueError
            If the selected asic_num and pixel_num are not found in the spectra

        Returns
        -------
        spectrum : Spectrum1D
        """
        pixel_ind = (self.asics == asic_num) * (self.pixel_nums == pixel_num)
        if np.sum(pixel_ind) != 1:
            raise ValueError(f"asic {asic_num} and {pixel_num} not found.")
        int_spec = np.sum(self.specs[:, pixel_ind, :].data, axis=0)[0]
        # the spectral axis is at the center of the bins
        result = Spectrum1D(
            flux=u.Quantity(int_spec, "count"),
            spectral_axis=u.Quantity(self.bins, "pix"),
            uncertainty=StdDevUncertainty(np.sqrt(int_spec) * u.count),
        )
        return result

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.specs[key]
        elif isinstance(key, slice):
            if isinstance(key.start, Time) and isinstance(key.stop, Time):
                ind = (self.time > key.start) * (self.time < key.stop)
                return type(self)(
                    self.pkt_spec[ind], self.specs[ind, :, :], self._pixel_ids
                )
        return self


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
