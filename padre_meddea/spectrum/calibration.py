"""Tools to analyze spectra"""

import numpy as np
import matplotlib.pyplot as plt

import astropy.units as u

import specutils
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region

from padre_meddea.spectrum.spectrum import PhotonList, SpectrumList
import padre_meddea.util.util as util

specutils.conf.do_continuum_function_check = False

BA_LINE_ENERGIES = [7.8, 11.8, 30.85, 35, 53.5, 57.8, 81] * u.keV


def get_calfunc_barium_rough(spec: Spectrum1D, plot: bool = False):
    """
    Given a full range Ba-133 spectrum, return a rough linear calibration function
    by finding and fitting only the two strongest lines (30.85 keV, 81 keV).
    It does this by splitting the spectrum into two regions and finding the
    maximum value in those regions. The saturated values at high energies are ignored.

    Parameters
    ----------
    spec : Spectrum1D
        The Ba-133 spectrum
    plot : bool
        If True, then display a plot of the spectrum with the line peaks found.
    Returns
    -------
    np.poly1d linear fit
    """
    # the two strongest lines in the spectrum 30.85, 81
    line_energies = u.Quantity([BA_LINE_ENERGIES[2], BA_LINE_ENERGIES[-1]])
    line_centers = np.zeros(len(line_energies))
    # split the spectrum into two regions, ignore the top end which includes saturated events.
    region_edges_percent = [0.17, 0.48, 0.73]
    region_edges_spec = np.floor(region_edges_percent * spec.spectral_axis.max())
    spec_regions = SpectralRegion(
        [
            [region_edges_spec[0], region_edges_spec[1]],
            [region_edges_spec[1], region_edges_spec[2]],
        ]
    )
    for i, (this_energy, this_region) in enumerate(zip(line_energies, spec_regions)):
        sub_spec = extract_region(spec, this_region)
        mind = np.argmax(sub_spec.data)
        line_centers[i] = sub_spec.spectral_axis[mind].value
    # fit rough calibration using just these two lines
    p = np.polyfit(line_energies.value, line_centers, 1)
    f = np.poly1d(p)
    if plot:
        plt.plot(spec.spectral_axis, spec.flux)
        for this_line in line_centers:
            plt.axvline(this_line)
    return f


def fit_peak_parabola(spec: Spectrum1D) -> float:
    """Given a spectral region with a single line, fit a parabola
    to the peak and return the position of the maximum

    Parameters
    ----------
    spec : Spectrum1D

    Returns
    -------
    peak_center : float
    """
    x = spec.spectral_axis.value
    y = spec.flux.value
    max_ind = np.argmax(y)
    # TODO add edge case for max at index value 0 or max index
    fit_x = [x[max_ind - 1], x[max_ind], x[max_ind + 1]]
    fit_y = [y[max_ind - 1], y[max_ind], y[max_ind + 1]]
    p = np.polyfit(fit_x, fit_y, 2)
    fit_peak = -p[1] / (2.0 * p[0])
    return fit_peak


def fit_peaks(
    spec: Spectrum1D,
    line_centers: u.Quantity,
    plot: bool = False,
    fit_func="parabola",
    window=30,
) -> u.Quantity:
    """
    Given a spectrum and a set of approximate peak or line centers,
    perform a fit for each and return the fitted peak location for each.

    Parameters
    ----------
    spec : Spectrum1D
        The input spectrum
    line_centers : u.Quantity
        The approximate location of the line peaks
    plot : bool
        If True, then plot the data and fit for each line center region.
    fit_func : str
        The fit function for finding the peak value.
    window : int
        Number of points to consider around the line center

    Returns
    -------
    fit_centers
    """
    fit_centers = line_centers.copy()
    spec_units = line_centers[0].unit
    fit_window_halfwidth = window * spec_units
    for i, this_line in enumerate(line_centers):
        line_region = SpectralRegion(
            this_line - fit_window_halfwidth, this_line + fit_window_halfwidth
        )
        sub_spec = extract_region(spec, line_region)
        fit_centers[i] = fit_peak_parabola(sub_spec) * spec_units
        if plot:
            plt.figure()
            plt.plot(sub_spec.spectral_axis.value, sub_spec.flux, label="data")
            plt.axvline(this_line.value, color="green", label="line_center")
            plt.axvline(fit_centers[i].value, color="red", label="fit peak")
            plt.legend()
            plt.show()
    return fit_centers


def calibrate_phlist_barium_linear(ph_list: PhotonList, plot: bool = False):
    """Given a PhotonList of a Ba-133 spectrum,
    perform a linear energy calibration for all detectors and pixels.

    Parameters
    ----------
    ph_list: PhotonList

    Returns
    -------
    lin_cal_params[num_asics,num_pixels,2]
        An array of linear calibration values for each pixel.
    """

    spec_bins = np.arange(0, 4097, 8, dtype=np.uint16) * u.pix
    lin_cal_params = np.zeros((4, 12, 2))
    all_pixels = util.PixelList.all()
    for i, this_pixel in enumerate(all_pixels):
        # fitting barium lines
        this_spec = ph_list.spectrum(pixel_list=this_pixel, bins=spec_bins)
        f = get_calfunc_barium_rough(this_spec)
        ba_line_centers = f(BA_LINE_ENERGIES.value)
        fit_line_centers = fit_peaks(
            this_spec, u.Quantity(ba_line_centers, this_spec.spectral_axis.unit)
        )
        if plot:
            plt.figure()
            plt.plot(this_spec.spectral_axis.value, this_spec.flux.value)
            for this_line, that_line in zip(fit_line_centers, ba_line_centers):
                plt.axvline(this_line, color="red", label="fit")
                plt.axvline(that_line, color="green", label="rough")
            plt.title(f"{this_spec['label'].value}")
            plt.show()
        # if this_pixel > 8:  # small pixel, remove the weak escape lines
        #    x = [fit_line_centers[0], fit_line_centers[1], fit_line_centers[-1]]
        #    y = [line_energies[0].value, line_energies[1].value, line_energies[-1].value]
        # else:
        x = fit_line_centers
        y = BA_LINE_ENERGIES.value
        p = np.polyfit(x, y, 1)
        f = np.poly1d(p)
        if plot:
            plt.figure()
            plt.plot(x, y, "x")
            plt.plot(x, f(x.value))
            plt.title(f"{this_spec['label'].value}")
            plt.show()
        lin_cal_params[this_pixel['asic'], this_pixel['pixel'], :] = p
    return lin_cal_params


def calibrate_speclist_barium_linear(spec_list: SpectrumList, plot: bool = False):
    """Given a PhotonList of a Ba-133 spectrum,
    perform a linear energy calibration for all detectors and pixels.

    Parameters
    ----------
    ph_list: PhotonList

    Returns
    -------
    lin_cal_params[num_asics,num_pixels,2]
        An array of linear calibration values for each pixel.
    """

    lin_cal_params = np.zeros((24, 2))
    for pixel_index, this_pixel in enumerate(spec_list.pixel_list):
        # fitting barium lines
        this_spec = spec_list.spectrum(pixel_list=this_pixel)
        f = get_calfunc_barium_rough(this_spec)
        # TODO: test lines for flux
        STRONG_BA_LINE_ENERGIES = [7.8, 30.85, 35, 81] * u.keV
        ba_line_centers = f(STRONG_BA_LINE_ENERGIES.value)
        fit_line_centers = fit_peaks(
            this_spec, u.Quantity(ba_line_centers, this_spec.spectral_axis.unit), window=5
        )
        if plot:
            plt.figure()
            plt.plot(this_spec.spectral_axis.value, this_spec.flux.value)
            for this_line, that_line in zip(fit_line_centers, ba_line_centers):
                plt.axvline(this_line, color="red", label="fit")
                plt.axvline(that_line, color="green", label="rough")
            plt.title(this_pixel['label'])
            plt.legend()
            plt.show()
        # if this_pixel > 8:  # small pixel, remove the weak escape lines
        #    x = [fit_line_centers[0], fit_line_centers[1], fit_line_centers[-1]]
        #    y = [line_energies[0].value, line_energies[1].value, line_energies[-1].value]
        # else:
        x = fit_line_centers
        y = STRONG_BA_LINE_ENERGIES.value
        p = np.polyfit(x, y, 1)
        f = np.poly1d(p)
        if plot:
            plt.figure()
            plt.plot(x, y, "x")
            plt.plot(x, f(x.value))
            plt.show()
        lin_cal_params[pixel_index, :] = p
    return lin_cal_params


def calibrate_linear_phlist(
    ph_list: PhotonList, lin_cal_params: np.array
) -> PhotonList:
    """Given an uncalibrated PhotonList and a complete set of linear calibration parameters
    produced by calibrate_phlist_barium_linear, apply the calibration to the
    PhotonList. Adds a new energy column.

    Paramters
    ---------
    Uncalibrated PhotonList

    Linear calibration parameter array

    Returns
    -------
    calibrated PhotonList
    """
    ph_list.event_list["energy"] = np.zeros(len(ph_list.event_list["atod"]))
    for this_asic in range(4):
        for this_pixel in range(12):
            ind = (ph_list.event_list["asic"] == this_asic) * (
                ph_list.event_list["pixel"] == this_pixel
            )
            cal_func = np.poly1d(lin_cal_params[this_asic, this_pixel, :])
            ph_list.event_list["energy"][ind] = cal_func(
                ph_list.event_list["atod"][ind]
            )
    return ph_list



def calibrate_linear_speclist(
    spec_list: SpectrumList, lin_cal_params: np.array
) -> SpectrumList:
    """Given an uncalibrated SpectrumList and a complete set of linear calibration parameters
    produced by calibrate_phlist_barium_linear, apply the calibration to the
    PhotonList. Adds a new energy column.

    Paramters
    ---------
    Uncalibrated PhotonList

    Linear calibration parameter array

    Returns
    -------
    calibrated SpectrumList
    """
    from scipy.interpolate import RectBivariateSpline
    new_spectral_axis = np.arange(0, 100, 0.1) * u.keV
    num_ts = spec_list.specs.shape[0]
    new_spec_data = np.zeros((num_ts, 24, len(new_spectral_axis)))
    this_y = (spec_list.time - spec_list.time[0]).to('s').value
    for i in range(24):
        f = np.poly1d(lin_cal_params[i, :])
        this_x = f(spec_list.specs[0, i].spectral_axis.value)
        z = spec_list.specs[:, i].data
        f2d = RectBivariateSpline(this_x, this_y, z.T)
        new_spec_data[:, i, :] = f2d(new_spectral_axis.value, this_y).T
    specs = Spectrum1D(
        spectral_axis=new_spectral_axis, flux=new_spec_data * u.ct
    )
    new_spec_list = SpectrumList(spec_list.pkt_list, specs, pixel_ids=spec_list._pixel_ids)

    return new_spec_list
