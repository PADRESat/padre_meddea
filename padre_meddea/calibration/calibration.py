"""
A module for all things calibration.
"""

from pathlib import Path
import tempfile

from astropy.io import fits, ascii
from astropy.time import Time
from astropy.table import Table

from swxsoc.util.util import create_science_filename

import padre_meddea
from padre_meddea import log
from padre_meddea.io import file_tools

# from padre_meddea.util.util import create_science_filename
from padre_meddea.io.file_tools import read_raw_file

__all__ = [
    "process_file",
    "get_calibration_file",
    "read_calibration_file",
    
    "spectrum",
    "calibrate_spectrum",
    "find_nearest",
    "gains_offsets"
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
    if filename.suffix == ".bin":
        parsed_data = read_raw_file(filename)
        if "photons" in parsed_data.keys():  # we have event list data
            ph_list = parsed_data["photons"]
            hdu = fits.PrimaryHDU(data=None)
            hdu.header["DATE"] = (Time.now().fits, "FITS file creation date in UTC")
            fits_meta = read_fits_keyword_file(
                padre_meddea._data_directory / "fits_keywords_primaryhdu.csv"
            )
            for row in fits_meta:
                hdu.header[row["keyword"]] = (row["value"], row["comment"])
            bin_hdu = fits.BinTableHDU(data=Table(ph_list))
            hdul = fits.HDUList([hdu, bin_hdu])

            output_filename = create_science_filename(
                "meddea",
                ph_list["time"][0].fits,
                "l1",
                descriptor="eventlist",
                test=True,
                version="0.1.0",
            )

            # Determine the temporary directory
            temp_dir = Path(tempfile.gettempdir())
            path = temp_dir / output_filename

            hdul.writeto(path, overwrite=overwrite)
            output_files = [path]

    #  calibrated_file = calibrate_file(data_filename)
    #  data_plot_files = plot_file(data_filename)
    #  calib_plot_files = plot_file(calibrated_file)

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


def read_fits_keyword_file(csv_file: Path):
    """Read csv file with default fits metadata information."""
    fits_meta_table = ascii.read(
        padre_meddea._data_directory / "fits_keywords_primaryhdu.csv",
        format="csv",
    )
    return fits_meta_table



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

    for this_spectra,this_pixel in zip(spectra,pixels): 
        if spec_plot: 
            plt.plot(this_spectra.spectral_axis, this_spectra.flux, label='Photon data')
            for this_roi in rois:
                plt.axvspan(this_roi[0], this_roi[1], alpha=0.5)
            plt.title(f'Pixel {this_pixel}')
            plt.show()
        fit = calibrate_spectrum(this_spectra, line_energies, roi=this_roi, plot=True)
        cal_offsets.append(fit.coef[0])
        cal_gains.append(fit.coef[1])
        energy_axis = fit(this_spectra.spectral_axis.value)
        plt.show()
        for line_energy in line_energies: 
            index, nearest_energy = find_nearest(energy_axis, value=line_energy)
            adc_channels.append(this_spectra.spectral_axis.value[index])
            energies.append(nearest_energy)
    return adc_channels, energies, cal_offsets, cal_gains
