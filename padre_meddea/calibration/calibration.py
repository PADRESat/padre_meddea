"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

import numpy as np

from astropy.io import fits
from astropy.time import Time
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

from scipy.signal import find_peaks

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
    "energy_cal"]


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





# include tools for calibration. 
# do not define functions that operate on files or filenames. 
# these functions should only operate on objects...
# what about plotting functions? I have complicated plotting routines that work on arrays/lists... 

# testing updated spectrum plotting tools. 

def get_spec(event_list, asic, pixel): 
    """
    Reads the contents of an event list and returns the baseline-subtracted energy spectrum (in ADC channel space). 

    Parameters
    ----------
    event_list: Timeseries object
        Photon event list. 
    asic: int
        ASIC number.
    pixel: int
        Pixel number. 

    Returns
    -------
    spectrum: Spectrum1D object
        Baseline-subtracted energy spectrum. 
    """
    # slice the event_list: 
    sliced_list=event_list[(event_list['asic']==asic) & (event_list['pixel']==pixel)]
    # baseline subtraction: 
    energy=np.array(sliced_list['atod'], dtype='float64')
    baseline=np.array(sliced_list['baseline'], dtype='float64')
    data, bins=np.histogram(((energy-baseline)+np.mean(baseline)), bins=np.arange(0,2**12-1))
    spectrum=Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, 'pix'))
    return spectrum

def get_spec_arr(asics, pixels, event_list):
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
            this_spectrum=get_spec(event_list=event_list, asic=this_asic, pixel=this_pixel)
            spectra[this_asic].append(this_spectrum)
    return spectra

# for each ASIC, plot the spectra on top of each other. 
def plot_spec(asics, pixels, spectra, hdu, save=False):
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
        plt.suptitle(f'{hdu.filename()}')
        if save==True: 
            plt.savefig(f'{hdu.filename()}_{this_asic}_plot.png')
        plt.show()

def plot_subspec(asics, pixels, spectra, hdu, save=False):
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

    hdu: fits header data unit.

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
        if save==True: 
            plt.savefig(f'{hdu.filename()}_{this_asic}.png')
        plt.show()

def find_rois(spectrum, prominence, width, distance):
    '''
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

    '''
    line_centers, properties=find_peaks(spectrum.data, prominence=prominence, width=width, distance=distance)
    rois = []
    for j in range(len(properties["widths"])):
        roi = [properties["left_ips"][j],properties["right_ips"][j]]
        rois.append(roi)
    return line_centers, rois

def cal_spec(spectrum, line_centers=None, rois=None, plot=None):
    """
    Takes a spectrum object and the centroids of its spectral lines (in energy space) and returns a function that converts from ADC Channel space to energy space. 

    Parameters
    ----------
    spectrum: Spectrum1D object

    line_centers_keV: arr
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

# global variable.
# move this further up. 
ba133_line_centers=[30.973, 34.920, 80]

# energy calibration, lpix
def energy_cal(asics, pixels, spectra, prominence, width, distance):
    '''
    Parameters: 
    '''
    for this_asic in asics:
        for this_pixel in pixels[0:8]:
            #spectrum=get_spec(event_list=event_list, asic=this_asic, pixel=this_pixel)
            spectrum=spectra[this_asic][this_pixel]

            spectrum.flux[0:800]=[0*u.ct for i in spectrum.spectral_axis[0:800]]
            spectrum.flux[3800:4200]=[0*u.ct for i in spectrum.spectral_axis[3800:4200]]

            line_centers, rois=find_rois(spectrum, prominence=200, width=10, distance=100) # hard-coded for now
            fit=cal_spec(spectrum, line_centers=ba133_line_centers, rois=rois, plot=True)
        plt.title(f'ASIC {this_asic}, Lpix')
        plt.show()