import matplotlib.pyplot as plt
from roentgen.absorption import Material
import astropy.units as u
import numpy as np
import padre_meddea.io.amptek as amptek
import pandas as pd

'''
This module contains the tools needed to analyze the transmissions of the Al and Be windows. 
'''

__all__ = [
    "plot_energy_spec",
    "plot_trans",
    "stats",
    "window_meas_results"]

'''
Input
    unfilt_spec: measured unfiltered spectrum. 
    filt_spec: measured filtered spectrum. 
Output
    A plot showing the measured unfiltered and filtered spectra. 
'''
def plot_energy_spec(unfilt_spec, filt_spec): 
    fig, ax=plt.subplots(layout="constrained")
    energy_axis=unfilt_spec.meta['calib'](unfilt_spec.spectral_axis.value) 
    idx=(energy_axis > 1)*(energy_axis < 20)
    energy_axis=energy_axis[idx]   

    fig.suptitle('Measured Spectrum')
    ax.plot(energy_axis, unfilt_spec.flux[idx]/unfilt_spec.meta['realtime'], label=unfilt_spec.meta['filename'])
    ax.plot(energy_axis, filt_spec.flux[idx]/filt_spec.meta['realtime'], label=filt_spec.meta['filename'])
    ax.legend(loc='lower right')
    ax.set_yscale("log")
    ax.grid()
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Counts/s")
    plt.show()

'''
Input
    unfilt_spec: measured unfiltered spectrum. 
    filt_spec: measured filtered spectrum. 
    material: the element that the filter is composed of; string.
    thickness: the thickness of the material (in mm); decimal value. 
Output
    A plot showing the measured and modeled transmission. 
'''
def plot_trans(unfilt_spec, filt_spec, material, thickness): 
    fig, ax=plt.subplots(layout="constrained")

    energy_axis=unfilt_spec.meta['calib'](unfilt_spec.spectral_axis.value)    
    mat=Material(material, thickness=thickness*u.mm)
    idx=(energy_axis > 1)*(energy_axis < 20)

    transmission=mat.transmission(energy_axis[idx]*u.keV)
    filename=filt_spec.meta['filename']
    ax.plot(energy_axis[idx], (filt_spec.flux[idx]/filt_spec.meta['realtime'])/(unfilt_spec.flux[idx]/unfilt_spec.meta['realtime']), label=filt_spec.meta['filename']) # the integration time for the unfiltered measurements was nearly double that of the filtered measurements; could that be why the transmission is nearly twice what we expect it to be? divide out by the realtimes.   
    ax.plot(energy_axis[idx], transmission, label=f'{mat.name} {thickness} mm')
    
    fig.suptitle(f'{mat.name} {thickness} mm Transmission')
    ax.grid()
    ax.set_ylabel("Transmission")
    ax.set_xlabel("Energy (keV)")
    ax.set_xlim(0, 20)
    #ax.set_ylim(0, 1.2)
    ax.legend()
    plt.savefig(f'{filename}_transmission.png')
    plt.show()

'''
Input
    spectra: an array of spectra. 

Output
    acc_times: accumulation times
    tot_cts: measured count rates (fast and slow channels)
    counts: measured counts from the Gross Area (slow channel). 
    uncertainties: square root of counts. 
    count_rates: Gross Area divided by the Accumulation Time. 
    uncertainties_ct_rt: square root of the Gross Area divided by the Accumulation Time. 
'''
def stats(spectra): 
    acc_times = []
    tot_cts = []
    cts = []
    ct_rts = []
    uncertainties = []
    uncertainties_ct_rt = []
    for this_spectrum in spectra: 
        acc_time=this_spectrum.meta['realtime']
        acc_times.append(acc_time.value)

        tot_ct=(this_spectrum.flux).sum()
        tot_cts.append(tot_ct.value)
    
        ct=(this_spectrum[181*u.pix:1872*u.pix].flux).sum() 
        cts.append(ct.value)

        uncertainty=np.sqrt(ct) 
        uncertainties.append(uncertainty.value)

        ct_rt=ct/this_spectrum.meta['realtime'] 
        ct_rts.append(ct_rt.value)

        uncertainty_ct_rt=uncertainty/this_spectrum.meta['realtime'] 
        uncertainties_ct_rt.append(uncertainty_ct_rt.value)

    return acc_times, tot_cts, cts, ct_rts, uncertainties, uncertainties_ct_rt

'''
Input
    unfilt1: data file; unfiltered measurement for windows 1 and 3. 
    unfilt2: data file; unfiltered measurement for windows 4 and 2. 
    files: data files; an array filtered measurements for windows 1, 3, 4, and 2. 

Output
    plots: energy and transmission spectra for each measurement. 
    files: a .csv file contianing the accumulation times, total counts, gross counts, count rates, uncertainties, and count rate uncertainties for each measurement. 
'''
def window_meas_results(unfilt1, unfilt2, files, material, thickness, stats_filename): 
    # convert the raw data to spectra. 
    unfilt1_spec=amptek.read_mca(unfilt1)
    unfilt2_spec=amptek.read_mca(unfilt2)
    spectra=[]
    for this_filename in files: 
        spectrum=amptek.read_mca(this_filename)
        spectra.append(spectrum)
    
    # plot the transmission. 
    for this_spectrum in spectra[0:2]: 
        plot_energy_spec(unfilt1_spec, this_spectrum)
        plot_trans(unfilt1_spec, this_spectrum, material=material, thickness=thickness)

    for this_spectrum in spectra[2:4]: 
        plot_energy_spec(unfilt2_spec, this_spectrum)
        plot_trans(unfilt2_spec, this_spectrum, material=material, thickness=thickness)

    # export stats to file
    df = pd.DataFrame({
        'spectrum': files,
        'accumulation time': np.around(stats(spectra)[0],4),
        'total counts': np.around(stats(spectra)[1],4),
        'counts (gross area)': np.around(stats(spectra)[2],4),
        'count_rates (gross area / accumulation time)': np.around(stats(spectra)[3],4),
        'uncertainties (sqrt(gross area))': np.around(stats(spectra)[4],4),
        'uncertainties_count_rate (sqrt(gross area) / accumulation time)':np.around(stats(spectra)[5],4)})
    df.to_csv(f'{stats_filename}.csv', index=False)