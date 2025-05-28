"""Tools to analyze spectra"""

import numpy as np
import matplotlib.pyplot as plt

import specutils
from specutils import Spectrum1D
from specutils.manipulation import extract_region
from specutils import SpectralRegion

from astropy.modeling import models
import astropy.units as u

specutils.conf.do_continuum_function_check = False

from padre_meddea.calibration.spectrum import PhotonList


def find_rough_cal(spec: Spectrum1D, plot: bool = False):
    """
    Given a Ba-133 spectrum, return a rough linear calibration function
    by finding and fitting only the 30.85 keV line and 81 keV line.
    It does this by splitting the spectrum into two regions and finding the
    maximum value in those regions.

    Parameters
    ----------
    spec : Spectrum1D
        The Ba-133 spectrum
    plot : bool
        If True, then display a plot of the spectrum with the line identified.
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


def fit_peaks_para(
    spec: Spectrum1D, line_centers: u.Quantity, plot: bool = False
) -> u.Quantity:
    """
    Given a spectrum and a set of line approximate peaks,
    fit a parabola to the line peak region and return the fit value.

    Parameters
    ----------
    spec : Spectrum1D
        The input spectrum
    line_centers : u.Quantity
        The approximate location of the line peaks
    plot : bool
        If True, then plot the data and fit for each line center region.

    Returns
    -------
    fit_centers
    """
    fit_centers = line_centers.copy()
    spec_units = line_centers[0].unit
    fit_window_halfwidth = 30 * spec_units
    for i, this_line in enumerate(line_centers):
        line_region = SpectralRegion(
            this_line - fit_window_halfwidth, this_line + fit_window_halfwidth
        )
        sub_spec = extract_region(spec, line_region)
        p = np.polyfit(sub_spec.spectral_axis.value, sub_spec.flux.value, 2)
        fit_peak = -p[1] / (2.0 * p[0])
        fit_centers[i] = fit_peak * spec_units
        if plot:
            plt.figure()
            plt.plot(sub_spec.spectral_axis, sub_spec.flux, label="data")
            f = np.poly1d(p)
            plt.plot(sub_spec.spectral_axis, f(sub_spec.spectral_axis), label="fit")
            plt.axvline(this_line, color="green", label="line_center")
            plt.axvline(fit_peak, color="red", label="fit peak")
            plt.legend()
    return fit_centers


def calibrate_phlist_barium_linear(ph_list: PhotonList, plot: bool = False):
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
            fit_line_centers = fit_peaks_para(this_spec, ba_line_centers)
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
                plt.plot(x, f(x))
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
