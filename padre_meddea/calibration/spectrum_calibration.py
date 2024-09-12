'''
This module provides the tools necessary to calibrate a spectrum. 
'''

from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt
from specutils.spectra import SpectralRegion
from specutils import Spectrum1D
from specutils.fitting import fit_lines
from specutils.spectra import SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import estimate_line_parameters
import padre_meddea
import padre_meddea.io.file_tools as file_tools
import datetime as dt
from astropy.modeling import models
from astropy.timeseries import aggregate_downsample, TimeSeriesd
import astropy.units as u
from scipy import stats
from scipy.interpolate import UnivariateSpline
import ccsdspy
from ccsdspy import PacketField, PacketArray
from ccsdspy.utils import (
    split_packet_bytes,
    count_packets,
    split_by_apid,
    read_primary_headers,
)

__all__ = ["spectrum", "calibrate_spectrum", "nearest_energy", "find_nearest"]

def spectrum(filename: Path, time_range=None, channel=None, bins=None):
    """
    Extract and record the spectra from a given file. 

    Parameters
    ----------
    filename: Path
        A file to read.
    time_range
        An array of the start and end times. 
    channel
        ASIC channel number. 
    bins
        Number of histogram bins.
        
    Returns
    -------
    result
        data, bins
            An array of spectra arranged in ascending pixel order. 

    Examples
    --------
    """
    this_ph_list = file_tools.parse_ph_packets(filename)
    if channel:
        this_ph_list = this_ph_list[this_ph_list['asic_channel'] == channel]
    if time_range:
        this_ph_list = this_ph_list.loc[time_range[0]:time_range[1]]
    if bins is None:
        bins = np.arange(0, 2**12-1)
    data, bins = np.histogram(this_ph_list['atod'], bins=bins)
    result = Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, "pix"), uncertainty=np.sqrt(data))
    result = Spectrum1D(flux=result.flux, spectral_axis=result.spectral_axis,uncertainty=None)
    return result

def calibrate_spectrum(spectrum, line_centers, roi=None, plot=False):
    """

    Given line energies for each region of interest, calibrate the spectrum.

    Parameters
    ----------
    spectrum:
        A spectrum to calibrate.
    line_centers
        An array of the centroids of the spectral lines.  
    roi
        An array of the regions of interest (ROI). Each ROI is an array of the start and end points in ADC Channel space.
    plot
        Allows for plotting of the spectral region with ROIs overlaid.
        
    Returns
    -------
    result.convert()
        Polynomial fit for calibration. 

    Examples
    --------
    """
    means = []
    for this_roi in roi:
        this_region = SpectralRegion(this_roi[0] * u.pix, this_roi[1] * u.pix)
        sub_spectrum = extract_region(spectrum, this_region)
        params = estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init = models.Gaussian1D(amplitude=params.amplitude, mean=params.mean, stddev=params.stddev)
        g_fit = fit_lines(sub_spectrum, g_init)
        means.append(g_fit.mean.value)
    result = Polynomial.fit(means, line_centers, deg=1)
    if plot:
        plt.plot(means, line_centers, "x")
        plt.plot(means, result(np.array(means)), "-", label=f"{result.convert()}")
        plt.ylabel("Energy [keV]")
        plt.xlabel("Channel")
        plt.legend()
    return result.convert()

def find_nearest(array, value):
    '''
    Find the values in an array nearest to the corresponding values in another array. 

    Parameters
    ----------
    array
        An array of values. 
    value
        A value to evaluate. 
    Returns
    -------
    idx, array[idx]

    Examples
    --------
    Find the values in energy space [keV] nearest to the corresponding values in ADC [ADC Ch] space. 
    '''
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def gains_offsets(spectrum, line_energies, spec_plot=False, energy_plot=False): 
    '''
    Calibrate the energy axis by transforming from ADC Channel space to Energy space. 

    Parameters
    ----------
    spectrum
        A spectrum to evaluate.
    line_energies
        Energies of the calibration lines in keV. 
    spec_plot
        Plot the uncalibrated spectrum. 
    energy_plot
        Plot the calibrated spectrum.
        
    Returns
    -------
    adc_channels, energies, cal_offsets, cal_gains
        List of ADC Channels nearest to the specified line_energies, corresponding energies, offsets and gains from the energy calibration. 

    Examples
    --------
    
    '''
    adc_channels = []
    energies = []
    cal_offsets = []
    cal_gains = []

    for i in range(len(pixels)): 
        if spec_plot: 
            plt.plot(spectra[i].spectral_axis, spectra[i].flux, label='Photon data')
            for j in ba133_rois[i]:
                plt.axvspan(j[0], j[1], alpha=0.5)
            plt.title(f'Pixel {pixels[i]}')
            plt.show()
        fit = calibrate_spectrum(spectra[i], line_energies, roi=ba133_rois[i], plot=True)
        cal_offsets.append(fit.coef[0])
        cal_gains.append(fit.coef[1])
        energy_axis = fit(spectra[i].spectral_axis.value)
        plt.show()
        for line_energy in line_energies: 
            index, nearest_energy = find_nearest(energy_axis, value=line_energy)
            adc_channels.append(spectra[i].spectral_axis.value[index])
            energies.append(nearest_energy)
    return adc_channels, energies, cal_offsets, cal_gains
