"""
A module for all things calibration.
"""

import os
from pathlib import Path
import tempfile

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.time import Time
from astropy.table import Table
from astropy.timeseries import TimeSeries

from specutils import Spectrum1D
import astropy.units as u
from specutils.manipulation import extract_region
from astropy.modeling import models
from specutils.fitting import estimate_line_parameters
from numpy.polynomial import Polynomial
from specutils.fitting import fit_lines
from specutils.spectra import SpectralRegion
from scipy.signal import find_peaks

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools
from padre_meddea.util import util, validation
import padre_meddea.io.aws_db as aws_db

from padre_meddea.util.util import create_science_filename, calc_time
from padre_meddea.io.file_tools import read_raw_file
from padre_meddea.io.fits_tools import (
    get_primary_header,
    get_obs_header,
    get_comment,
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
    "light_curve",
    "timing"
]


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

    if file_path.suffix.lower() in [".bin", ".dat"]:
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
            pkt_list, event_list = parsed_data["photons"]
            log.info(
                f"Found photon data, {len(event_list)} photons and {len(pkt_list)} packets."
            )
            aws_db.record_photons(pkt_list, event_list)

            event_list = Table(event_list)
            event_list.remove_column("time")

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(file_path, "l1", "photon")

            for this_keyword in ["DATE-BEG", "DATE-END", "DATE-AVG"]:
                primary_hdr[this_keyword] = (
                    event_list.meta.get(this_keyword, ""),
                    get_comment(this_keyword),
                )
            primary_hdr["DATEREF"] = (primary_hdr["DATE-BEG"], get_comment("DATEREF"))

            path = create_science_filename(
                "meddea",
                time=primary_hdr["DATE-BEG"],
                level="l1",
                descriptor="eventlist",
                test=True,
                version="0.1.0",
            )
            primary_hdr["FILENAME"] = (path, get_comment("FILENAME"))

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            pkt_list = Table(pkt_list)
            pkt_list.remove_column("time")
            pkt_header = get_obs_header()
            pkt_hdu = fits.BinTableHDU(pkt_list, header=pkt_header, name="PKT")
            pkt_hdu.add_checksum()
            hit_header = get_obs_header()
            hit_hdu = fits.BinTableHDU(event_list, header=hit_header, name="SCI")
            hit_hdu.add_checksum()
            hdul = fits.HDUList([empty_primary_hdu, hit_hdu, pkt_hdu])

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

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(file_path, "l1", "housekeeping")

            date_beg = calc_time(hk_data["timestamp"][0])
            primary_hdr["DATEREF"] = (date_beg.fits, get_comment("DATEREF"))

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

            path = create_science_filename(
                "meddea",
                time=date_beg,
                level="l1",
                descriptor="hk",
                test=True,
                version="0.1.0",
            )
            primary_hdr["FILENAME"] = (path, get_comment("FILENAME"))

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)

            # Create HK HDU
            hk_header = get_obs_header()
            hk_hdu = fits.BinTableHDU(data=hk_table, header=hk_header, name="HK")
            hk_hdu.add_checksum()

            # add command response data if it exists  in the same fits file
            cmd_header = get_obs_header()
            if parsed_data["cmd_resp"] is not None:
                data_ts = parsed_data["cmd_resp"]
                cmd_header["DATEREF"] = (
                    data_ts.time[0].fits,
                    get_comment("DATEREF"),
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
                cmd_hdu = fits.BinTableHDU(
                    data=data_table, header=cmd_header, name="READ"
                )
                cmd_hdu.add_checksum()
            else:  # if None still end an empty Binary Table
                cmd_hdu = fits.BinTableHDU(data=None, header=cmd_header, name="READ")
            hdul = fits.HDUList([empty_primary_hdu, hk_hdu, cmd_hdu])

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)
        if parsed_data["spectra"] is not None:
            ts, spectra, ids = parsed_data["spectra"]
            aws_db.record_spectra(ts, spectra, ids)
            asic_nums, channel_nums = util.parse_pixelids(ids)
            # asic_nums = (ids & 0b11100000) >> 5
            # channel_nums = ids & 0b00011111
            # TODO check that asic_nums and channel_nums do not change

            # Get FITS Primary Header Template
            primary_hdr = get_primary_header(file_path, "l1", "spectrum")

            dates = {
                "DATE-BEG": ts.time[0].fits,
                "DATE-END": ts.time[-1].fits,
                "DATE-AVG": ts.time[len(ts.time) // 2].fits,
            }
            primary_hdr["DATEREF"] = (dates["DATE-BEG"], get_comment("DATEREF"))
            for this_keyword, value in dates.items():
                primary_hdr[this_keyword] = (
                    value,
                    get_comment(this_keyword),
                )

            path = create_science_filename(
                "meddea",
                time=dates["DATE-BEG"],
                level="l1",
                descriptor="spec",
                test=True,
                version="0.1.0",
            )
            primary_hdr["FILENAME"] = (path, get_comment("FILENAME"))

            spec_header = get_obs_header()
            spec_hdu = fits.ImageHDU(data=spectra.data, header=spec_header, name="SPEC")
            spec_hdu.add_checksum()

            data_table = Table()
            data_table["pkttimes"] = ts["pkttimes"]
            data_table["pktclock"] = ts["pktclock"]
            data_table["asic"] = asic_nums
            data_table["channel"] = channel_nums
            data_table["seqcount"] = ts["seqcount"]

            pkt_header = get_obs_header()
            pkt_hdu = fits.BinTableHDU(data=data_table, header=pkt_header, name="PKT")
            pkt_hdu.add_checksum()

            empty_primary_hdu = fits.PrimaryHDU(header=primary_hdr)
            hdul = fits.HDUList([empty_primary_hdu, spec_hdu, pkt_hdu])

            # Set the temp_dir and overwrite flag based on the environment variable
            if lambda_environment:
                temp_dir = Path(tempfile.gettempdir())  # Set to temp directory
                overwrite = True  # Set overwrite to True
                path = temp_dir / path

            hdul.writeto(path, overwrite=overwrite)
            output_files.append(path)

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
    Reads the contents of an event list and returns the photon spectrum (in ADC channel space). 

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
    Reads the contents of an event list and returns an array of spectra. 
    
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
        plt.show()

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
        plt.show()

# depricated
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
        params=estimate_line_parameters(sub_spectrum, models.Gaussian1D()) # TBD: define an initial set of parameters.
        g_init=models.Gaussian1D(amplitude=params.amplitude, mean=params.mean, stddev=params.stddev)
        g_fit=fit_lines(sub_spectrum, g_init) # TBD: this should include the uncertainties on the peak positions; check. There will be uncertainty on the gain, which will include uncertainty from the peak positions themselves. 
        #print(g_fit)
        means.append(g_fit.mean.value)
    result=Polynomial.fit(means, line_centers, deg=1)
    if plot:
        plt.plot(means, line_centers, "x")
        plt.plot(means, result(np.array(means)), "-", label=f"{result.convert()}")
        plt.ylabel("Energy [keV]")
        plt.xlabel("Channel")
        plt.legend()
    return result.convert()

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