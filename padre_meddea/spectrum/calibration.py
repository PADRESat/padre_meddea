"""Tools to analyze spectra"""

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

from astropy.modeling import models
import astropy.units as u

import specutils
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import estimate_line_parameters, fit_lines

from padre_meddea.spectrum.spectrum import PhotonList

specutils.conf.do_continuum_function_check = False


def find_rough_cal(spec: Spectrum1D, plot: bool = False):
    """
    Given a Ba-133 spectrum, return a rough linear calibration function
    by finding and fitting only the two strongest lines (30.85 keV, 81 keV).
    It does this by splitting the spectrum into two regions and finding the
    maximum value in those regions.

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
    line_energies = [30.85, 81] * u.keV
    line_centers = np.zeros(len(line_energies))
    spec_regions = SpectralRegion([[700, 2000] * u.pix, [2000, 3000] * u.pix])
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
    fit_x = [x[max_ind-1], x[max_ind], x[max_ind+1]]
    fit_y = [y[max_ind-1], y[max_ind], y[max_ind+1]]
    p = np.polyfit(fit_x, fit_y, 2)
    fit_peak = -p[1] / (2.0 * p[0])
    return fit_peak


def fit_peaks(
    spec: Spectrum1D, line_centers: u.Quantity, plot: bool = False, fit_func='parabola', window=30
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
    print(line_centers.unit)
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
        print(fit_centers)
    return fit_centers


def calibrate_barium_linear(ph_list: PhotonList, plot: bool = False):
    """Given a PhotonList of a Ba-133 spectrum,
    perform a linear energy calibration for all detectors and pixels.

    Parameters
    ----------
    ph_list: PhotonList

    Returns
    -------
    lin_cal_params[4,12,2]
        An array of linear calibration values for each pixel
    """
    ba_line_energies = [7.8, 11.8, 30.85, 35, 53.5, 57.8, 81] * u.keV
    spec_bins = np.arange(0, 4097, 8, dtype=np.uint16)
    lin_cal_params = np.zeros((4, 12, 2))
    for this_asic in range(4):
        for this_pixel in range(12):
            # fitting barium lines
            this_spec = ph_list.spectrum(
                asic_num=this_asic, pixel_num=this_pixel, bins=spec_bins
            )
            f = find_rough_cal(this_spec)
            ba_line_centers = f(ba_line_energies.value)
            fit_line_centers = fit_peaks_para(this_spec, u.Quantity(ba_line_centers, this_spec.spectral_axis.unit))
            if plot:
                plt.figure()
                plt.plot(this_spec.spectral_axis.value, this_spec.flux.value)
                for this_line, that_line in zip(fit_line_centers, ba_line_centers):
                    plt.axvline(this_line, color="red", label="fit")
                    plt.axvline(that_line, color="green", label="rough")
                plt.title(f"{this_asic} {this_pixel}")
                plt.show()
            # if this_pixel > 8:  # small pixel, remove the weak escape lines
            #    x = [fit_line_centers[0], fit_line_centers[1], fit_line_centers[-1]]
            #    y = [line_energies[0].value, line_energies[1].value, line_energies[-1].value]
            # else:
            x = fit_line_centers
            y = ba_line_energies.value
            p = np.polyfit(x, y, 1)
            f = np.poly1d(p)
            if plot:
                plt.figure()
                plt.plot(x, y, "x")
                plt.plot(x, f(x.value))
                plt.title(f"asic {this_asic} pixel {this_pixel}")
                plt.show()
            lin_cal_params[this_asic, this_pixel, :] = p
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
    return result.convert()  # return the function
