import re

import numpy as np
from scipy import interpolate

import astropy.units as u
from astropy.table import QTable, Table
from astropy.time import Time
from astropy.io import ascii
from astropy.modeling.models import Gaussian1D, custom_model
from astropy.constants.codata2018 import h, c, e, k_B

from scipy.stats.sampling import NumericalInversePolynomial

import padre_meddea

# the following are used to calculate escape lines from the Caliste-SO detector
cd_kalpha1 = 23173.6 * u.eV  # x-ray data booklet 2009
te_kalpha1 = 27472.3 * u.eV  # x-ray data booklet 2009

# the expected count rate at the peak of an X-class flare
# for all detectors combined
# TODO: provide better justification for this number. Calculation can be found in proposal.
XCLASS_FLARE_RATE = 22421.0 * u.s**-1

#  TODO: add escape lines to the ba133_lines Table

__all__ = [
    "barium_spectrum",
    "flare_spectrum",
    "setup_phgenerator_ba",
    "setup_phgenerator_flare",
    "get_flare_rate",
]

# load flare spectrum file
flare_spectrum_data = QTable(
    ascii.read(
        padre_meddea._data_directory / "SOL2002-07-23_RHESSI_flare_spectrum.csv",
        format="csv",
    )
)
for this_col in flare_spectrum_data.colnames:
    unit_str = re.findall(r"\(.*?\)", this_col)[0][1:-1]
    flare_spectrum_data[this_col].unit = u.Unit(unit_str)

flare_timeseries = QTable(
    ascii.read(
        padre_meddea._data_directory / "SOL2002-07-23_GOESXRS_lightcurve.csv",
        format="csv",
    )
)
flare_timeseries["sec_from_start"].unit = u.s

ba133_lines = QTable(
    ascii.read(
        padre_meddea._data_directory / "ba133.csv",
        format="csv",
    )
)
ba133_lines["energy (eV)"].unit = u.eV


@u.quantity_input
def barium_spectrum(fwhm: u.keV):
    """Provide the spectrum model from radioactive Ba133 as observed by the
    Caliste-SO detector. It includes Cd and Te escape lines.

    Returns
    -------
    model : ~astropy.modeling.models

    #TODO: Add filter at low energies, add fall off due to detector thickness
    """

    for i, this_line in enumerate(ba133_lines):
        cd_escape_line = this_line["energy (eV)"] - cd_kalpha1
        te_escape_line = this_line["energy (eV)"] - te_kalpha1
        if i == 0:
            spec = Gaussian1D(
                amplitude=this_line["intensity"],
                mean=this_line["energy (eV)"],
                stddev=fwhm,
            )
        else:
            spec += Gaussian1D(
                amplitude=this_line["intensity"],
                mean=this_line["energy (eV)"],
                stddev=fwhm,
            )
        if (
            cd_escape_line > 0 * u.eV
        ):  # TODO: need to scale down the intensity of line based on probably of escape
            spec += Gaussian1D(
                amplitude=this_line["intensity"], mean=cd_escape_line, stddev=fwhm
            )
        if (
            te_escape_line > 0 * u.eV
        ):  # TODO: need to scale down the intensity of line based on probably of escape
            spec += Gaussian1D(
                amplitude=this_line["intensity"], mean=te_escape_line, stddev=fwhm
            )
    return spec


@custom_model
def flare_spectrum(x: u.keV):
    """Provides the average spectrum of an X4.8 flare in x-rays.

    The minimum energy available is 3.5 keV and the highest is 14950 keV.

    Parameters
    ----------
    factor : int
        The factor to scale the flare

    Returns
    -------
    model : ~astropy.modeling.models
    """
    factor = 1
    # NOTE: it may be better to interpolate the log of the spectrum for high accuracy
    func = interpolate.interp1d(
        flare_spectrum_data["Bin mean (keV)"].value,
        np.log10(flare_spectrum_data[flare_spectrum_data.colnames[-1]].value * factor),
        bounds_error=True,
        assume_sorted=True,
    )
    return 10 ** func(x)


@custom_model
def get_flare_rate(x: u.s):
    """Provides a times series of an X4.8 flare in x-rays as measured by GOES
    XRS B (1 to 8 angstrom). The flare is clipped to limit flux at times when
    the power is less than 1e-5.

    x parameter : time in seconds
    y parameter : counts/s/detector

    Returns
    -------
    model : ~astropy.modeling.models
    """
    y = flare_timeseries["xrsb"]
    # remove the pre and post-flare times
    y[y < 1e-5] = 1e-30
    # normalize this lightcurve to counts/s/det, from proposal calculation
    y = y / y.max() * XCLASS_FLARE_RATE.value
    func = interpolate.interp1d(
        flare_timeseries["sec_from_start"],
        np.log10(y),
        bounds_error=True,
        assume_sorted=True,
    )
    return 10 ** func(x)


@u.quantity_input
def setup_phgenerator_ba(fwhm: u.keV):
    """Setup the random number generate to create random photons from Ba133,
    the calibration source.

    Note that this can take a few minutes to run.

    Parameters
    ----------
    fwhm : full-width-half-maximum
        Provide the resolution of the detector system.

    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    ba = barium_spectrum(fwhm)

    class spec:
        def pdf(self, x):
            return ba(x * u.keV)

    dist = spec()
    urng = np.random.default_rng(seed=42)
    rng = NumericalInversePolynomial(dist, random_state=urng, domain=[1, 150])

    return rng


def gen_random_ba_photons(ba_rvs, num):
    """Generate random photons from Ba133 source

    Returns
    -------
    photons : ~np.array
    """
    ba_rvs = ba_rvs.rvs(num)
    return ba_rvs


@u.quantity_input
def setup_phgenerator_flare(factor, filename=None):
    """Setup the random number generate to create random photons from Ba133.

    Note that this can take a few minutes to run.

    Parameters
    ----------
    factor : float
        The strength of the flare relative to X class.
    filename : Path
        If set, saves the random generator object to a file for future use.
    Returns
    -------
    generator : ~scipy.stats.sampling.NumericalInversePolynomial
    """
    fa = flare_spectrum()

    class spec:
        def pdf(self, x):
            return fa(x * u.keV)

    dist = spec()
    urng = np.random.default_rng(seed=42)
    rng = NumericalInversePolynomial(dist, random_state=urng, domain=[5, 150])
    return rng


def next_ph_time(rate: u.s**-1, size):
    """Generate a list of photon wait times based on a photon rate.

    rate : u.Quantity
        The photon count rate.
    size : int
        The number of wait times to generate.

    Note: using numpy is faster then python expovariate

    Returns
    -------
    result : ~np.array
    """
    return -np.log(1 - np.random.random(size=size)) / rate


def generate_calib_ph_list(
    rate: u.s**-1, num: int, width: u.keV, output_file: bool = False
):
    """Generate a list of random photon generated from the calibration source.

    Parameters
    ----------
    rate : u.Quantity
        The photon count rate.
    size : int
        The number of photons to generate.
    width : u.Quantity
        The resolution of the detector or the width of the lines.
    output_file : bool
        If True then save the list into a csv file.

    Returns
    -------
    result : ~astropy.table.Qtable
    """
    ph_wait_times = next_ph_time(rate, num)
    ph_arrival_times = ph_wait_times.cumsum()

    # generate random energies
    ba_rng = setup_phgenerator_ba(width)
    ph_energies = ba_rng.rvs(num)
    ph_origin_label = np.chararray(num, itemsize=5)
    ph_origin_label[:] = "calib"

    det_num, pixel_num = get_random_det_pixel(num)

    ph_table = QTable(
        [
            ph_arrival_times,
            ph_wait_times,
            ph_energies,
            det_num,
            pixel_num,
            ph_origin_label,
        ],
        names=(
            "times (s)",
            "wait times (s)",
            "energy (keV)",
            "detnum",
            "pixelnum",
            "source",
        ),
        meta={"name": "simulated photons list of calib source"},
    )

    if output_file:
        ph_table.write("simul_photon_list.csv", format="ascii.csv", overwrite=True)

    return ph_table


def generate_flare_ph_list(goes_class: str, num: int, output_file: bool = False):
    """Generate a list of random photon generated at the peak of a flare peak.
    The flare rate does not change.

    Parameters
    ----------
    goes_class : str
        The GOES flare class. Accepted values are A, B, C, M, X.
    size : int
        The number of photons to generate.
    output_file : bool
        If True then save the list into a csv file.

    Returns
    -------
    result : ~astropy.table.Qtable
    """
    RATE_CONV_FACTOR = {"X": 1, "M": 1e-1, "C": 1e-2, "B": 1e-3, "A": 1e-4}
    rate = XCLASS_FLARE_RATE * RATE_CONV_FACTOR[goes_class.upper()]

    ph_wait_times = next_ph_time(rate, num)
    ph_arrival_times = ph_wait_times.cumsum()

    # generate random energies
    fl_rng = setup_phgenerator_flare(0.1)
    ph_energies = fl_rng.rvs(num)

    ph_energies = fl_rng.rvs(num)
    ph_origin_label = np.chararray(num, itemsize=5)
    ph_origin_label[:] = "flare"

    # generate random detector and pixel assignments

    det_num, pixel_num = get_random_det_pixel(num)

    ph_table = QTable(
        [
            ph_arrival_times,
            ph_wait_times,
            ph_energies,
            det_num,
            pixel_num,
            ph_origin_label,
        ],
        names=(
            "times (s)",
            "wait times (s)",
            "energy (keV)",
            "detnum",
            "pixelnum",
            "source",
        ),
        meta={"name": f"simulated photons list at peak of  {goes_class}-class flare"},
    )

    if output_file:
        ph_table.write("simul_photon_list.csv", format="ascii.csv", overwrite=True)

    return ph_table


def generate_photon_list_file(output_file=True):
    """Generate a simulated photon list using a flare light curve.
    The flux before and after the flare is dominated by the calibration
    spectrum.

    WARNING: This is a slow process.
    """
    # TODO: Update this to make use of generate_calib_ph_list

    #  flare_rng = setup_phgenerator_flare(1)
    time = np.arange(10001) * u.s

    ba_rate = 1.0 / u.s  # count / s
    ba_ph_num = int(time.max() * ba_rate)

    # over 10000 s we should expect 10000 counts
    # let's generate 10000 wait times,

    ba_ph_wait_times = next_ph_time(ba_rate, ba_ph_num)
    ba_ph_arrival_times = ba_ph_wait_times.cumsum()

    max_flare_ph_num = int(1e8)
    fl_ph_wait_times = np.zeros(max_flare_ph_num) * u.s
    fl_ph_arrival_times = np.zeros(max_flare_ph_num) * u.s
    ph_counter = 0
    rate_update_times = 10 * u.s
    for this_time in np.arange(0, time.max().value, rate_update_times.value):
        this_flare_rate = get_flare_rate()(this_time * u.s) / u.s
        # expected number of flare counts but add margin since it could be greater
        fl_ph_num = int(rate_update_times * this_flare_rate) * 2
        print(f"{fl_ph_num} {this_time}")

        if fl_ph_num > 1000:  # ignore flare nums that are too low in the interval
            this_fl_ph_wait_times = next_ph_time(this_flare_rate, fl_ph_num)
            this_fl_ph_arrival_times = this_fl_ph_wait_times.cumsum()
            # keep only the photons inside the time interval
            index = this_fl_ph_arrival_times < rate_update_times
            print(index.sum())
            flare_ph_num = index.sum()
            print(f"Got {flare_ph_num} flare photons.")

            fl_ph_wait_times[
                ph_counter : ph_counter + flare_ph_num
            ] = this_fl_ph_wait_times[index]
            fl_ph_arrival_times[ph_counter : ph_counter + flare_ph_num] = (
                this_fl_ph_arrival_times[index] + this_time * u.s
            )
            ph_counter += flare_ph_num

    # remove zeros
    nonzero_index = fl_ph_wait_times > 0
    fl_ph_wait_times = fl_ph_wait_times[nonzero_index]
    fl_ph_arrival_times = fl_ph_arrival_times[nonzero_index]
    fl_ph_num = np.sum(nonzero_index)

    # generate random energies
    ba_rng = setup_phgenerator_ba(1 * u.keV)
    ba_ph_energies = ba_rng.rvs(ba_ph_num)
    fl_rng = setup_phgenerator_flare(0.1)
    fl_ph_energies = fl_rng.rvs(fl_ph_num)

    ph_wait_times = np.concatenate((fl_ph_wait_times, ba_ph_wait_times))
    ph_arrival_times = np.concatenate((fl_ph_arrival_times, ba_ph_arrival_times))
    ph_energies = np.concatenate((fl_ph_energies, ba_ph_energies))
    ph_origin_label = np.chararray(fl_ph_num + ba_ph_num, itemsize=5)
    ph_origin_label[0:fl_ph_num] = "flare"
    ph_origin_label[fl_ph_num + 1 : fl_ph_num + ba_ph_num] = "ba133"

    sort_index = np.argsort(ph_arrival_times)

    det_num, pixel_num = get_random_det_pixel(len(sort_index))

    ph_table = QTable(
        [
            ph_arrival_times[sort_index],
            ph_wait_times[sort_index],
            ph_energies[sort_index],
            det_num,
            pixel_num,
            ph_origin_label[sort_index],
        ],
        names=(
            "times (s)",
            "wait times (s)",
            "energy (keV)",
            "detnum",
            "pixelnum",
            "source",
        ),
        meta={"name": "simulated photons list including X-class flare and ba133"},
    )
    if output_file:
        ph_table.write("simul_photon_list.csv", format="ascii.csv", overwrite=True)

    return ph_table


def get_random_det_pixel(num):
    """Generate random detector and pixel assignments assuming even
    illumintation on all detectors and pixels.

    Returns
    -------
    detector_numbers, pixel_numbers
    """
    det_number = np.random.randint(4, size=num)

    pixel_choices = np.arange(padre_meddea.NUM_PIXELS)
    weights = padre_meddea.NUM_LARGE_PIXELS * [
        padre_meddea.RATIO_TOTAL_LARGE_TO_SMALL_PIX / padre_meddea.NUM_LARGE_PIXELS
    ]
    weights += padre_meddea.NUM_SMALL_PIXELS * [
        (1 - padre_meddea.RATIO_TOTAL_LARGE_TO_SMALL_PIX)
        / padre_meddea.NUM_SMALL_PIXELS
    ]

    pixel_nums = np.random.choice(pixel_choices, p=weights, size=num)

    return det_number, pixel_nums
