"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

import numpy as np

from astropy.io import fits
from astropy.time import Time
import astropy.timeseries as timeseries
import datetime
from astropy.table import Table
import astropy.units as u
from astropy.modeling import models

import numpy as np
from numpy.polynomial import Polynomial
import matplotlib.pyplot as plt

from specutils import Spectrum1D
from specutils.manipulation import extract_region
from specutils.fitting import estimate_line_parameters
from specutils.fitting import fit_lines
from specutils.spectra import SpectralRegion
from specutils import analysis

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy import signal

import pandas as pd

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools, fits_tools
from padre_meddea.util import util, validation
import padre_meddea.io.aws_db as aws_db

from padre_meddea.util.util import create_science_filename, calc_time
from padre_meddea.io.file_tools import read_raw_file
from padre_meddea.io.fits_tools import (
    add_process_info_to_header,
    get_primary_header,
    get_std_comment,
)

__all__ = [
    "process_file",
    "get_calibration_file",
    "read_calibration_file",

    "get_spec",
    "get_spec_arr",
    "plot_spec",
    "plot_subspec",
    "find_rois",
    "cal_spec",
    "auto_cal",
    "gauss_fit",
    "plot_lightcurve",
    "timing"]


def process_file(filename: Path, overwrite=False) -> list:
    """
    This is the entry point for the pipeline processing.
    It runs all of the various processing steps required.

    Parameters
    ----------
    data_filename: str
        Fully specificied filename of an input file

    Returns
    -------
    output_filenames: list
        Fully specificied filenames for the output files.
    """
    log.info(f"Processing file {filename}.")
    # Check if the LAMBDA_ENVIRONMENT environment variable is set
    lambda_environment = os.getenv("LAMBDA_ENVIRONMENT")
    output_files = []
    file_path = Path(filename)

    if file_path.suffix == ".bin":
        # Before we process, validate the file with CCSDS
        custom_validators = [validation.validate_packet_checksums]
        validation_findings = validation.validate(
            file_path,
            valid_apids=list(padre_meddea.APID.values()),
            custom_validators=custom_validators,
        )
        for finding in validation_findings:
            log.warning(f"Validation Finding for File : {filename} : {finding}")

        parsed_data = read_raw_file(file_path)
        if parsed_data["photons"] is not None:  # we have event list data
            event_list, pkt_list = parsed_data["photons"]
            primary_hdr = get_primary_header()
            primary_hdr = add_process_info_to_header(primary_hdr)
            primary_hdr["LEVEL"] = (0, get_std_comment("LEVEL"))
            primary_hdr["DATATYPE"] = ("event_list", get_std_comment("DATATYPE"))
            primary_hdr["ORIGAPID"] = (
                padre_meddea.APID["photon"],
                get_std_comment("ORIGAPID"),
            )
            primary_hdr["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))

            for this_keyword in ["DATE-BEG", "DATE-END", "DATE-AVG"]:
                primary_hdr[this_keyword] = (
                    event_list.meta.get(this_keyword, ""),
                    get_std_comment(this_keyword),
                )

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            pkt_list = Table(pkt_list)
            pkt_list.remove_column("time")
            pkt_hdu = fits.BinTableHDU(pkt_list, name="PKT")
            pkt_hdu.add_checksum()
            hit_hdu = fits.BinTableHDU(event_list, name="SCI")
            hit_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary_hdu, hit_hdu, pkt_hdu])

            path = create_science_filename(
                instrument="meddea",
                time=primary_hdr["DATE-BEG"],
                level="l1",
                descriptor="eventlist",
                test=True,
                version="0.1.0",
            )

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            # Write the file, with the overwrite option controlled by the environment variable
            hdul.writeto(path, overwrite=overwrite)
            # Store the output file path in a list
            output_files.append(path)
        if parsed_data["housekeeping"] is not None:
            hk_data = parsed_data["housekeeping"]
            # send data to AWS Timestream for Grafana dashboard
            aws_db.record_housekeeping(hk_data)
            hk_table = Table(hk_data)

            primary_hdr = get_primary_header()
            primary_hdr = add_process_info_to_header(primary_hdr)
            primary_hdr["LEVEL"] = (0, get_std_comment("LEVEL"))
            primary_hdr["DATATYPE"] = ("housekeeping", get_std_comment("DATATYPE"))
            primary_hdr["ORIGAPID"] = (
                padre_meddea.APID["housekeeping"],
                get_std_comment("ORIGAPID"),
            )
            primary_hdr["ORIGFILE"] = (file_path.name, get_std_comment("ORIGFILE"))

            date_beg = calc_time(hk_data["timestamp"][0])
            primary_hdr["DATEREF"] = (date_beg.fits, get_std_comment("DATEREF"))

            hk_table["seqcount"] = hk_table["CCSDS_SEQUENCE_COUNT"]
            colnames_to_remove = [
                "CCSDS_VERSION_NUMBER",
                "CCSDS_PACKET_TYPE",
                "CCSDS_SECONDARY_FLAG",
                "CCSDS_SEQUENCE_FLAG",
                "CCSDS_APID",
                "CCSDS_SEQUENCE_COUNT",
                "CCSDS_PACKET_LENGTH",
                "CHECKSUM",
                "time",
            ]
            for this_col in colnames_to_remove:
                if this_col in hk_table.colnames:
                    hk_table.remove_column(this_col)

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hk_hdu = fits.BinTableHDU(data=hk_table, name="HK")
            hk_hdu.add_checksum()

            # add command response data if it exists
            if parsed_data["cmd_resp"] is not None:
                data_ts = parsed_data["cmd_resp"]
                this_header = fits.Header()
                this_header["DATEREF"] = (
                    data_ts.time[0].fits,
                    get_std_comment("DATEREF"),
                )
                aws_db.record_cmd(data_ts)
                data_table = Table(data_ts)
                colnames_to_remove = [
                    "CCSDS_VERSION_NUMBER",
                    "CCSDS_PACKET_TYPE",
                    "CCSDS_SECONDARY_FLAG",
                    "CCSDS_SEQUENCE_FLAG",
                    "CCSDS_APID",
                    "CCSDS_SEQUENCE_COUNT",
                    "CCSDS_PACKET_LENGTH",
                    "CHECKSUM",
                    "time",
                ]
                for this_col in colnames_to_remove:
                    if this_col in hk_table.colnames:
                        data_table.remove_column(this_col)
                cmd_hdu = fits.BinTableHDU(data=data_table, name="READ")
                cmd_hdu.add_checksum()
            else:  # if None still end an empty Binary Table
                this_header = fits.Header()
                cmd_hdu = fits.BinTableHDU(data=None, header=this_header, name="READ")
            hdul = fits.HDUList([empty_primary_hdu, hk_hdu, cmd_hdu])

            path = create_science_filename(
                instrument="meddea",
                time=date_beg,
                level="l1",
                descriptor="hk",
                test=True,
                version="0.1.0",
            )

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)
        if parsed_data["spectra"] is not None:
            spec_data = parsed_data["spectra"]

    # add other tasks below
    return output_files


def raw_to_l0(filename: Path):
    if not (filename.suffix == "bin"):
        raise ValueError(f"File {filename} extension not recognized.")

    data = file_tools.read_raw_file(filename)


def get_calibration_file(time: Time) -> Path:
    """
    Given a time, return the appropriate calibration file.

    Parameters
    ----------
    data_filename: str
        Fully specificied filename of the non-calibrated file (data level < 2)
    time: ~astropy.time.Time

    Returns
    -------
    calib_filename: str
        Fully specificied filename for the appropriate calibration file.

    Examples
    --------
    """
    return None


def read_calibration_file(calib_filename: Path):
    """
    Given a calibration, return the calibration structure.

    Parameters
    ----------
    calib_filename: str
        Fully specificied filename of the non-calibrated file (data level < 2)

    Returns
    -------
    output_filename: str
        Fully specificied filename of the appropriate calibration file.

    Examples
    --------
    """

    # if can't read the file
    return None



def get_spec(event_list, asic, pixel, step, baseline_sub=False): 
    """
    Reads the contents of an event list and returns the energy spectrum (in ADC channel space). 

    Parameters
    ----------
    event_list: Timeseries object
        Photon event list. 

    asic: int
        ASIC number. If set to None, returns spectrum across all asics. 

    pixel: int
        Pixel number. If set to None, returns spectrum across all pixels.

    step: int
        Binning step size. 

    baseline_sub: Boolean
        If set to True, subtracts the baseline. If set to False, does not subtract the baseline.

    Returns
    -------
    spectrum: Spectrum1D object
        Baseline-subtracted energy spectrum. 
    """

    sliced_list=event_list[(event_list['asic']==asic) & (event_list['pixel']==pixel)]
    if asic==None: 
        sliced_list=event_list[(event_list['pixel']==pixel)]
    if pixel==None:
        sliced_list=event_list[(event_list['asic']==asic)]
    if (asic==None) & (pixel==None):
        sliced_list=event_list
    if baseline_sub==True: 
        energy=np.array(sliced_list['atod'], dtype='float64')
        baseline=np.array(sliced_list['baseline'], dtype='float64')
        data, bins=np.histogram(((energy-baseline)+np.mean(baseline)), bins=np.arange(0,2**12-1,step))
    else: 
        data, bins=np.histogram(sliced_list['atod'], bins=np.arange(0,2**12-1,step))
    spectrum=Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, 'pix'))
    return spectrum

def get_spec_arr(asics, pixels, event_list, step, baseline_sub):
    """
    Reads the contents of an event list and returns a multi-dimensional array of spectra. 
    
    Parameters
    ----------
    asics: arr
        Array of ASICs to iterate over. We typically use asics=np.arange(4).

    pixels: arr
        Array of pixels to iterate over. We typically use pixels=np.arange(12). 

    event_list: Timeseries object
        Photon event list.

    Returns
    -------
    spectra: arr
        Multi-dimensional array of spectra. Each element of this array corresponds to an ASIC and is itself an array of spectra, each corresponding to a pixel. 
    """
    spectra=[]
    for this_asic in asics: 
        spectra.append([])
        for this_pixel in pixels: 
            if baseline_sub==True: 
                this_spectrum=get_spec(event_list, asic=this_asic, pixel=this_pixel, step=step, baseline_sub=True)
                spectra[this_asic].append(this_spectrum)
            else:
                spectra[this_asic].append(this_spectrum)
    return spectra

# for each ASIC, plot the spectra on top of each other.
def plot_spec(asics, pixels, spectra, save=False):
    """
    Plots the measured spectra for each pixel, for each ASIC in a series of individual plots (in ADC channel space). 

    Parameters
    ----------
    asics: arr
        Array of asic numbers. Default is 1-4. 

    pixels: arr
        Array of pixel numbers. Default is 1-12. 
    
    spectra: arr
        Multi-dimensional array of spectra. Each element of the array (which corresponds to a an ASIC), contains 12 spectra, each corresponding to a pixel.

    hdu: fits header data unit.

    Returns
    -------
    """    
    for this_asic in asics: 
        for this_spectrum, this_pixel in zip(spectra[this_asic], pixels):
            plt.plot(this_spectrum.spectral_axis, this_spectrum.flux, label=f'Pixel: {this_pixel}')
        plt.legend()
        plt.title(f'ASIC {this_asic}')
        #plt.suptitle(f'{hdu.filename()}')
        #if save==True: 
        #    plt.savefig(f'{hdu.filename()}_{this_asic}_plot.png')
        plt.show()

# plot spectra in nidividual subplots.
def plot_subspec(asics, pixels, spectra, save=False):
    """
    Plots the measured spectra for each pixel, for each ASIC in a series of subplots (in ADC channel space). 

    Parameters
    ----------
    asics: arr
        Array of asic numbers. Default is 1-4. 

    pixels: arr
        Array of pixel numbers. Default is 1-12. 
    
    spectra: arr
        Multi-dimensional array. Each element of the array corresponds to a an ASIC and contains 12 spectra, each corresponding to a pixel. Each spectrum is itself a Spectrum1D object.

    Returns
    -------
    """
    for this_asic in asics:
        fig, ax = plt.subplots(3, 4, figsize=(10,7))
        for this_pixel, i in zip(pixels, np.arange(pixels.shape[0])):
            plt.subplot(3, 4, i+1)
            spectrum=spectra[this_asic][this_pixel]
            plt.plot(spectrum.spectral_axis, spectrum.flux)
            plt.title(f"Pixel {this_pixel}")
        fig.tight_layout()
        plt.suptitle(f'ASIC {this_asic}')
        #plt.suptitle(f'{hdu.filename()}')
        #if save==True: 
        #    plt.savefig(f'{hdu.filename()}_{this_asic}.png')
        plt.show()

# this works, but not reliably. 
# inputs to "find_peaks" must be hard-coded, which is not good practice.
def find_rois(spectrum, prominence, width, distance):
    """
    Find the regions of interest (ROIs) in ADC Channel space over which to perform the energy calibration. 

    Parameters
    ----------
    spectrum: Spectrum1D object

    prominence: 
    
    width: 

    distance: 

    Returns
    -------
    line_centers: 

    rois: 

    """
    line_centers, properties=find_peaks(spectrum.data, prominence=prominence, width=width, distance=distance)
    rois = []
    for j in range(len(properties["widths"])):
        roi = [properties["left_ips"][j],properties["right_ips"][j]]
        rois.append(roi)
    return line_centers, rois

# trying something new...
'''
def cal_spec(spectrum, line_centers=None, rois=None, plot=None):
    """
    Takes a spectrum object and the centroids of its spectral lines (in energy space) and returns a function that converts from ADC Channel space to energy space. 

    Parameters
    ----------
    spectrum: Spectrum1D object

    line_centers: arr
        Centroids of the spectral lines in energy space. 

    Returns
    -------
    result: function
        Function that converts from ADC Channel space to energy space. 
    """
    means=[]
    for this_roi in rois:
        this_region=SpectralRegion(this_roi[0] * u.pix, this_roi[1] * u.pix)
        sub_spectrum=extract_region(spectrum, this_region)
        params=estimate_line_parameters(sub_spectrum, models.Gaussian1D())
        g_init=models.Gaussian1D(amplitude=params.amplitude, mean=params.mean, stddev=params.stddev)
        g_fit=fit_lines(sub_spectrum, g_init)
        means.append(g_fit.mean.value)
    result=Polynomial.fit(means, line_centers, deg=1)
    if plot:
        plt.plot(means, line_centers, "x")
        plt.plot(means, result(np.array(means)), "-", label=f"{result.convert()}")
        plt.ylabel("Energy [keV]")
        plt.xlabel("Channel")
        plt.legend()
    return result.convert()
'''

def cal_spec(line_centers, line_energies, plot=False):
    '''
    Parameters

    line_centers: arr
        Locations of centers of lines in ADC channel space. 

    line_energies: arr
        Locations of centers of lines in energy space. 

    plot: Boolean
        if True, plots the calibration fit. 

    Returns
    
    result: function
        Fitting function. 
    '''
    result=Polynomial.fit(line_centers, line_energies, deg=1)
    if plot:
        plt.plot(line_centers, line_energies, "x")
        plt.plot(line_centers, result(line_centers), "-", label=f"{result.convert()}")
        plt.ylabel("Energy [keV]")
        plt.xlabel("Channel")
        plt.legend()
    return result.convert()


def auto_cal(raw_spectra, sm_spectra, asics, pixels, line_centers, plot=False):
    '''
    Parameters

    raw_spectra: arr
        Array of raw spectra.
    sm_spectra: arr
        Array of smoothed spectra. The Gaussian filter works on this array. 
    asics: arr
        Array of ASICs. 
    pixels: arr
        Array of pixels. 
    line_centers: arr
        Locations of the spectral lines used for calibration in energy space. 

    Returns: 
    gain: arr
        Array of coefficients. 

    '''
    gain=[] #save the coefficients from the energy calibration.
    for this_asic in asics: 
        gain.append([])
        for this_raw_spectrum, this_sm_spectrum, this_pixel in zip(raw_spectra[this_asic], sm_spectra[this_asic], pixels):
            filtered_spec=gaussian_filter1d(this_sm_spectrum.flux, sigma=0.1)
            tMax=signal.argrelmax(filtered_spec)[0]
            tMin=signal.argrelmin(filtered_spec)[0]
            if plot==True: 
                plt.plot(this_sm_spectrum.flux, label = 'raw')
                #plt.plot(filtered_spec, label = 'filtered')
                plt.plot(tMax, filtered_spec[tMax], 'o', mfc= 'none', label = 'max')
                plt.legend()
                #plt.show()
            line_centers=np.array(this_sm_spectrum.spectral_axis[tMax].value)
            #print(line_centers)
            fit=cal_spec(line_centers=line_centers, line_energies=ba133_line_centers, plot=False)
            #print(fit)
            gain[this_asic].append(fit)
            #plt.plot(fit(this_raw_spectrum.spectral_axis.value), this_raw_spectrum.flux)
            #for this_line_energy in ba133_line_centers: 
            #    plt.axvline(this_line_energy, color='grey', linestyle='dashed', alpha=0.05)
        #plt.title(f'ASIC {this_asic}')
        #plt.show()
    return gain


def gauss_fit(spectrum, fit, line_centers, rois): 
    """
    Parameters
    ----------
    spectrum: Spectrum1D object
        Spectrum to be fitted. 

    fit: function
        Function that converts from ADC Channel space to energy space

    line_centers: array
        Centers of lines to be fitted in ADC Channel space. 

    rois: array
        Regions of interest in ADC channel space. 

    Returns
    -------
    means: array
        Mean values of the Gaussian fit. 

    stddevs: array
        Standard deviations from the Gaussian fit. 

    fwhms: array
        FWHM values from the Gaussian fit. 

    fwhms2: array
        FWHM values from 

    amplitudes: array 

    """
    means = []
    stddevs = []
    fwhms = [] # from specutils.fitting.fit_lines (the Gaussian fit).
    fwhms2 = [] # from specutils.analysis.fwhm. 
    amplitudes=[]
    for this_line_center, this_roi in zip(line_centers, rois):
        #roi_lower=this_roi[0] # uncomment if you want to use the full ROI, not just the upper RHS.
        roi_lower=roi_lower=this_line_center # uncomment if you want to use the upper RHS of the ROI. 
        roi_upper=this_roi[1]
        
        this_region=SpectralRegion(roi_lower*u.pix, roi_upper*u.pix) # in ADC space
        sub_spec=extract_region(spectrum, this_region)
        params=estimate_line_parameters(sub_spec, models.Gaussian1D())
        g_init=models.Gaussian1D(amplitude=params.amplitude, mean=params.mean, stddev=params.stddev)
        g_fit=fit_lines(sub_spec, g_init)  
        means.append(fit(g_fit.mean.value))
        stddevs.append(fit(g_fit.stddev.value))
        amplitudes.append(g_fit.amplitude.value) 

        fwhm=fit(g_fit.mean.value+g_fit.fwhm.value/2.)-fit(g_fit.mean.value-g_fit.fwhm.value/2.) # "fit" cannot be applied to a width; it must be applied to individual values.
        fwhms.append(fwhm)
        # alternatively, calculate the fwhm from the Gaussian fit (using analysis.fwhm).
        fwhm2=2*(fit(analysis.fwhm(sub_spec).value+analysis.fwhm(sub_spec).value/2.)-fit(analysis.fwhm(sub_spec).value-analysis.fwhm(sub_spec).value/2.)) # "fit" cannot be applied to a width; it must be applied to individual values.
        fwhms2.append(fwhm2)

        energy_axis=fit(spectrum.spectral_axis.value)
        
        # plot the measured spectrum and the Gaussian fits for each spectral line on top.
        plt.plot(energy_axis, spectrum.flux)
        plt.axvspan(fit(roi_lower), fit(roi_upper), facecolor='green', alpha=0.1)
        plt.plot(energy_axis, g_fit(spectrum.spectral_axis), label=f'{fwhm}, {fwhm2}')
        plt.plot(fit(g_fit.mean.value), g_fit.amplitude.value, 'ro')
        plt.legend()
        plt.show()

        # plot the zoomed in spectrum and the Gaussian fits for each line on top.
        idx=(energy_axis>fit(roi_lower))*(energy_axis<fit(roi_upper))
        print(idx)
        plt.plot(energy_axis[idx], spectrum.flux[idx])
        plt.axvspan(fit(roi_lower), fit(roi_upper), facecolor='green', alpha=0.1)
        plt.plot(energy_axis[idx], g_fit(spectrum.spectral_axis[idx]), label=f'{fwhm}, {fwhm2}')
        plt.plot(fit(g_fit.mean.value), g_fit.amplitude.value, 'ro')
        plt.legend()
        plt.show()
    return means, stddevs, fwhms, fwhms2, amplitudes 


def plot_lightcurve(event_list, asics, pixels, int_time, energy_range, plot=False): 
    """
    Parameters
    ----------
    event_list: Timeseries object

    asics: array
        Array of asics to iterate over.

    pixels: array
        Array of pixels to iterate over.

    int_time: Integer
        Integration time in seconds.

    energy_range: array
        Range of energies to plot.
    
    plot: Boolean
        If True, then display a plot of lightcurves.

    Returns
    -------
        Array of lightcurves. 
        
    """
    # add a column to use to tally up the counts for the lightcurve.
    # remember to add a column to the oevent_list to use to tally up the counts to evaluate the lightcurve. 
    # event_list['count']=np.ones(len(event_list), dtype=np.uint8)  
    lightcurves=[]
    for this_asic in asics:
        lightcurves.append([])
        for this_pixel in pixels: 
            this_event_list=event_list[(event_list['asic']==this_asic) & (event_list['pixel']==this_pixel)]
            this_event_list=this_event_list[(this_event_list['atod']>=energy_range[0]) & (this_event_list['atod']<=energy_range[-1])] 
            this_lightcurve=timeseries.aggregate_downsample(time_series=this_event_list, time_bin_size=int_time, time_bin_start=this_event_list['time'][0], aggregate_func=np.sum)
            lightcurves[this_asic].append(this_lightcurve)
            if plot==True: 
                plt.plot(this_lightcurve['time_bin_start'].to_datetime(), this_lightcurve['count'])
                plt.title(f'ASIC {this_asic}m Pixel {this_pixel}')
                plt.xlabel('Time [m:s]')
                plt.ylabel(f'Counts [{int_time} bin]')
                plt.show()
    return lightcurves


def timing(event_list, plot_hist=False):
    """
    Take an event_list and find the average event rate and the minimum time interval between consecutive events. 

    Parameters
    ----------
    event_list: Timeseries object

    plot_hist: Boolean
        If True, plot the histogram of the unique times in the event_list.

    Returns
    -------
    
    """
    time=np.asarray([pd.to_datetime(l).timestamp() for l in np.unique(event_list['time'].value)]) # s
    deltat=np.gradient(np.unique(time))

    print(deltat)
    if plot_hist==True: 
        plt.hist(deltat, bins=50)
        #plt.title(f'{Time Gradient}')
        plt.ylim([0,1])
        plt.xlabel('Time Between Events [s]')
        plt.ylabel('# Events')
        plt.autoscale(enable=True, axis='both', tight=None)
        plt.show()
        
    avg_event_rate = np.around(1/np.average(deltat.mean()),3)
    print(f'Average Event Rate: {avg_event_rate} ct/s')   
    toto=np.unique(np.sort(deltat))
    min_delta_t=np.min(toto[toto!=0])*1E6
    print(f'Minimum nonzero time interval between two triggers: %4.2f µs'%(min_delta_t))
    return deltat, avg_event_rate, min_delta_t