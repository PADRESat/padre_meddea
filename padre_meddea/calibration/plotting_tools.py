"""
A module to plot/visualize photon data.
"""


import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

from specutils import Spectrum1D

__all__ = [
    "plot_agg",
    "plot_pix",
    "plot_asic",
    "plot_timeseries"
]

# make sure that the event_list has a column containing the calibrated energy values or else the 'energy' keyword will be meaningless.
# TODO: add support for basleine-subtracted spectra.
def plot_agg(event_list, asics, pixels, calibrated=False):
    """
    Plots the aggregate photon spectrum. 

    Parameters
    ----------
    event_list: Timeseries
        Photon event list.

    asics: arr
        Array of asic numbers.

    pixels: arr
        Array of pixel numbers.
    
    calibrated: Boolean
        If True, the plot will be in energy space.
        If False, the plot will be in ADC Channel space.
        
    Returns
        A plot showing the aggregate spectrum. 
    -------
    """
    if calibrated==True: 
        data, bins=np.histogram(event_list['energy'], bins=np.arange(0,100,0.1))
        plt.xlabel('Energy (keV)')
    else: 
        data, bins=np.histogram(event_list['atod'], bins=np.arange(0,2**12-1))
        plt.xlabel('ADC Channel')
    spectrum=Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, 'pix'))
    plt.plot(spectrum.spectral_axis, spectrum.flux)
    plt.title(f'Aggregate Spectrum')
    plt.ylabel('Counts')
    #filename=filename.rsplit('.', 1)[0]
    plt.legend()
    #plt.savefig(f'{filename}_agg_spec.png')
    plt.show()


# plot the spectra for each pixel.
def plot_pix(event_list, asics, pixels, calibrated=False): 
    """
    Plots the photon spectra for each pixel, for each asic. 

    Parameters
    ----------
    event_list: Timeseries
        Photon event list.

    asics: arr
        Array of asic numbers.

    pixels: arr
        Array of pixel numbers.
    
    calibrated: Boolean
        If True, the plots will be in energy space.
        If False, the plots will be in ADC Channel space.
        
    Returns
        Plots showing the photon spectra for each pixel, for each asic.
    -------
    """
    for this_asic in asics:
        fig, ax = plt.subplots(3, 4, figsize=(10,7))
        for this_pixel, i in zip(pixels, np.arange(pixels.shape[0])):
            plt.subplot(3, 4, i+1)
            sliced_list=event_list[(event_list['asic']==this_asic) & (event_list['pixel']==this_pixel)]
            if calibrated==True: 
                data, bins=np.histogram(sliced_list['energy'], bins=np.arange(0,100,0.1))
                plt.xlabel('Energy (keV)')
            else: 
                data, bins=np.histogram(sliced_list['atod'], bins=np.arange(0,2**12-1))
                plt.xlabel('ADC Channel')
            this_spectrum=Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, 'pix'))
            plt.plot(this_spectrum.spectral_axis, this_spectrum.flux)
            plt.title(f"Pixel {this_pixel}")
        fig.tight_layout()
        plt.suptitle(f'ASIC {this_asic}')
        plt.show()

# plot spectra by ASIC.
def plot_asic(event_list, asics, pixels, calibrated=False): 
    """
    Plots the photon spectra for each asic. 

    Parameters
    ----------
    event_list: Timeseries
        Photon event list.

    asics: arr
        Array of asic numbers.

    pixels: arr
        Array of pixel numbers.
    
    calibrated: Boolean
        If True, the plots will be in energy space.
        If False, the plots will be in ADC Channel space.
        
    Returns
        Plots showing the photon spectra for each asic.
    -------
    """
    fig, ax = plt.subplots(2, 2, figsize=(7,5))
    for this_asic, i in zip(asics, np.arange(asics.shape[0])):
        plt.subplot(2, 2, i+1)
        sliced_list=event_list[(event_list['asic']==this_asic)]
        if calibrated==True: 
            data, bins=np.histogram(sliced_list['energy'], bins=np.arange(0,100,0.1))
            plt.xlabel('Energy (keV)')
        else: 
            data, bins=np.histogram(sliced_list['atod'], bins=np.arange(0,2**12-1))
            plt.xlabel('ADC Channel')
        this_spectrum=Spectrum1D(flux=u.Quantity(data, 'count'), spectral_axis=u.Quantity(bins, 'pix'))
        plt.plot(this_spectrum.spectral_axis, this_spectrum.flux)
        plt.title(f'ASIC {this_asic}')
        plt.ylabel('Counts')
        fig.tight_layout()
    plt.show()

# plots the timeseries for each asic
def plot_timeseries(asics, event_list, calibrated=False):
    fig, ax = plt.subplots(2, 2, figsize=(7,7))
    for i, this_asic in enumerate(asics):
        plt.subplot(2,2,i+1)
        # define the event list to use. 
        this_event_list=event_list[(event_list['asic']==this_asic)]
        if calibrated==True: 
            plt.plot(this_event_list['time'].to_datetime(), this_event_list['energy'], ',')
            plt.ylabel('Energy (keV)')
        else: 
            plt.plot(this_event_list['time'].to_datetime(), this_event_list['atod'], ',')
            plt.ylabel('Energy ADC Channel')
        plt.title(f'ASIC {this_asic}')
        plt.xlabel('Time [m:s]')
        plt.ylabel(f'Energy [ADC Channel]') 
        fig.tight_layout()
    plt.show()