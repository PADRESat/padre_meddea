"""
This module provides io support for reading files created by the amptek detector.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import matplotlib.pyplot as plt

import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.io.fits import Header

from specutils import Spectrum1D
from specutils.spectra import SpectralRegion

from padre_meddea.util.util import str_to_fits_keyword


def read_mca(filename: Path, count_rate=False):
    """
    Read an amptek mca file.

    Parameters
    ----------
    filename: Path
        A file to read.
    count_rate: bool
        If True, then return data in count rate rather than counts.

    Returns
    -------
    spectrum: Spectrum1D
        All meta data with all capitals are direct from the mca file.
        Lower case metadata is added.

    Examples
    --------
    """
    with open(filename, "r", encoding="unicode_escape") as fp:
        in_data_section = False
        in_calib_section = False
        in_roi_section = False
        in_dp5config_section = False
        in_status_section = False
        in_spectrum_meta_section = False
        calib_data = {}
        roi_data = {}
        data = []
        line_number = 0
        hdr = Header()
        meta = {}
        for line in fp:
            #  figure out what section of the file we are in
            # the order used here is the order expected in the file
            # once in new section you are done with the previous section
            if line.count("PMCA SPECTRUM"):
                in_spectrum_meta_section = True
                continue
            if line.count("CALIBRATION"):
                in_calib_section = True
                in_spectrum_meta_section = False
                continue
            if line.count("ROI"):
                in_roi_section = True
                in_calib_section = False
                in_spectrum_meta_section = False
                continue
            if line.count("DATA"):
                in_data_section = True
                in_roi_section = False
                continue
            if line.count("END"):
                in_data_section = False
            if line.count("DP5 CONFIGURATION"):
                in_dp5config_section = True
                in_data_section = False
                continue
            if line.count("DPP STATUS"):
                in_dp5config_section = False
                in_status_section = True
                continue

            # now gather information
            if in_spectrum_meta_section:  # this is a measurement metadata tag
                keyword = line.split("-")[0]
                value = line.split("-")[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value))
            if in_calib_section:
                if line.count("-"):
                    continue
                else:
                    channel = float(line.split(" ")[0])
                    energy = float(line.split(" ")[1])
                    calib_data.update({channel: energy})
            if in_roi_section:
                roi_data.update({int(line.split(" ")[0]): int(line.split(" ")[1])})
            if in_data_section and not (line.count("<<") == 1):
                data.append(int(line))
            if in_dp5config_section and not (line.count("<<") == 1):
                keyword = line.split("=")[0].upper()
                value = line.split("=")[1].split(";")[0].strip()
                description = line.split("=")[1].split(";")[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value, description))
            if in_status_section and not (line.count("<<") == 1):
                keyword = line.split(":")[0]
                value = line.split(":")[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value))
            line_number += 1
        if count_rate:
            y = u.Quantity(np.array(data) / hdr["REALTIME"], "ct/s")
            uncertainty = StdDevUncertainty(
                u.Quantity(np.sqrt(data) / hdr["REALTIME"], "ct/s")
            )
        else:
            y = data * u.ct
            uncertainty = StdDevUncertainty(np.sqrt(data) * u.ct)

        if roi_data:
            rois = SpectralRegion(
                [[x1 * u.pix, x2 * u.pix] for x1, x2 in roi_data.items()]
            )

        meta.update({"header": hdr})

        meta.update({"roi": rois})
        meta.update({"livetime": hdr["LIVETIME"] * u.s})
        meta.update({"realtime": hdr["REALTIME"] * u.s})
        meta.update(
            {"obs_time": datetime.strptime(hdr["starttim"], "%m/%d/%Y %H:%M:%S")}
        )
        meta.update({"filename": filename})
        meta.update({"dtimfrac": 1 - hdr["livetime"] / hdr["realtime"]})
        meta.update({"rate": np.sum(data) * u.ct / meta["realtime"]})
        if len(calib_data) > 0:
            channels = list(calib_data.keys())
            energies = list(calib_data.values())
            meta.update({"calib": Polynomial.fit(channels, energies, deg=1)})

        spectrum = Spectrum1D(
            flux=y,
            spectral_axis=np.arange(len(data)) * u.pix,
            uncertainty=uncertainty,  # TODO add uncertainty, problem with ct/s as unit
            meta=meta,
        )

    return spectrum

def plot_amptek(amptek_spec, ax=None):
    """
    Plot an amptek spectrum.

    Parameters
    ----------
    amptek_spec: Spectrum1D
        All meta data with all capitals are direct from the mca file.
        Lower case metadata is added.

    Returns
    -------
        A plot showing the measured spectrum. 
    """
    if ax is None: 
        fig, ax=plt.subplots(layout="constrained")
    if isinstance(amptek_spec, list):
        for this_spec in amptek_spec:
            ax.plot(this_spec.spectral_axis, this_spec.flux/this_spec.meta['realtime'], label=amptek_spec.meta['filename'])
    else:
        ax.plot(amptek_spec.spectral_axis, amptek_spec.flux/amptek_spec.meta['realtime'], label=amptek_spec.meta['filename'])
        if amptek_spec.meta['roi']:
            for this_roi in amptek_spec.meta['roi']:
                ax.axvspan(this_roi.lower.value, this_roi.upper.value, alpha=0.2)
    ax.set_label("ADC Channel")

    if amptek_spec.meta['calib']:
        ax2 = ax.twiny()
        x = [
            amptek_spec.meta['calib'](amptek_spec.spectral_axis.value[0]),
            amptek_spec.meta['calib'](amptek_spec.spectral_axis.value[-1]),
        ]
        ax2.plot(x, [0, 0])
        ax2.set_label("Energy (keV)")
    ax.set_xlabel("ADC Channel")
    ax.set_ylabel("Counts/s")
    ax.set_yscale("log")
    ax.legend()

def plot_energy_spec(unfilt_spec, filt_spec, lower_bound, upper_bound, title): 
    """
    Plot an amptek spectrum that has not been attenuated (unfiltered) and an amptek spectrum that has been attenuated (filtered). 
    This is useful for visualizing measured spectra before calculating the transmission from the measured spectra.

    Parameters
    ----------
    unfilt_spec: Spectrum1D
        The unfiltered measurement. 
    
    filt_spec: Spectrum1D
        The filtered measurement. 

    lower_bound: int
        The minimum value of energy. 

    upper_bound: int
        The maximum value of energy. 

    title: string
        The title of the plot.

    Returns
    -------
        A plot showing the measured unfiltered and filtered spectra. 
    """
    fig, ax=plt.subplots(layout="constrained")
    energy_axis=unfilt_spec.meta['calib'](unfilt_spec.spectral_axis.value) 
    idx=(energy_axis>lower_bound)*(energy_axis<upper_bound)
    energy_axis=energy_axis[idx]   

    fig.suptitle(title)
    ax.plot(energy_axis, unfilt_spec.flux[idx]/unfilt_spec.meta['realtime'], label=unfilt_spec.meta['filename'])
    ax.plot(energy_axis, filt_spec.flux[idx]/filt_spec.meta['realtime'], label=filt_spec.meta['filename'])
    if unfilt_spec.meta['roi']:
            for this_roi in unfilt_spec.meta['roi']:
                ax.axvspan(unfilt_spec.meta['calib'](this_roi.lower.value), unfilt_spec.meta['calib'](this_roi.upper.value), alpha=0.2)
    ax.legend(loc='lower right')
    ax.set_yscale("log")
    ax.grid()
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts/s")
    ax.legend()
    plt.show()

def trans(unfilt_spec, filt_spec, mask=False):
    """
    Calculate the transmission function from measured data. 

    Parameters
    ----------
    unfilt_spec: Spectrum1D
        The unfiltered measurement. 
    
    filt_spec: Spectrum1D
        The filtered measurement. 

    mask: Boolean
        Replaces 0, nan, and inf values in the calculated transmission with finite, interpolated values. 
    """
    transmission=(filt_spec.flux/filt_spec.meta['realtime']) / (unfilt_spec.flux/unfilt_spec.meta['realtime'])
    if mask==True: 
        mask=((np.isnan(transmission)) | (transmission==0) | (transmission==np.inf)) # find where the transmission contains nan, 0, or infinite values.
        transmission[mask]=np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), transmission[~mask]) 
    return transmission