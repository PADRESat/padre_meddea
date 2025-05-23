"""Module to provide functions to calibrate housekeeping data"""

import numpy as np
from scipy.optimize import fsolve

from astropy.io import ascii
import astropy.units as u
from astropy.timeseries import TimeSeries
from astropy.table import Table, QTable

from padre_meddea import _package_directory

from .housekeeping import _data_directory, hk_definitions

calibration_table = ascii.read(_data_directory / "padre_meddea_calib_hk_0.csv")
calibration_table.add_index("name")


def get_calibration_func(hk_name: str) -> callable:
    """Given a housekeeping name, return a calibration function to convert from uncalibrated measurements to physical values.

    Parameters
    ----------
    hk_name: str

    Returns
    -------
    f: callable that returns physical quantities
    """
    f = _get_calibration_func(hk_name)
    unit_str = calibration_table.loc[hk_name]["unit_str"]

    def g(x):
        return u.Quantity(f(x), unit_str)

    return g


def _get_calibration_func(hk_name: str):
    """Given a housekeeping name, return the calibration function to convert from uncalibrated data values to physical values."""
    if hk_name in list(calibration_table["name"]):  # is it a housekeeping
        row = calibration_table.loc[hk_name]

        match row["cal_func"]:
            case "poly":
                params = [float(p) for p in row["cal_params"].split(",")]
                f = np.poly1d(params)
            case "log":
                params = [float(p) for p in row["cal_params"].split(",")]

                def f(x):
                    return params[0] * np.log(x) + params[1]

            case "bspline":
                f = fit_bspline(hk_name)
    else:
        raise ValueError(f"The housekeeping name, {hk_name}, is not recognized.")
    return f


def fit_bspline(hk_name: str) -> callable:
    from scipy.interpolate import make_interp_spline

    data = get_calibration_data(hk_name)
    ind = np.argsort(data["adc"])
    f = make_interp_spline(data["adc"][ind], data["value"][ind].value)

    def qf(x):
        return u.Quantity(f(x), data["value"].unit)

    return qf


def inverse_calibrate(hk_name: str, x: u.Quantity) -> float:
    """Given a housekeeping name, return the inverse function to convert from physical values to
    uncalibrated data values.

    Parameters
    ----------
    hk_name
        The housekeeping name
    x
        The physical value

    Return
    ------
    adc
        The ADC value that converts into the given physical value.
    """
    f = get_calibration_func(hk_name)

    def g(adc):
        return (f(adc) - x).value

    result = fsolve(g, 0.0)
    print(f"error = {x - get_calibration_func(hk_name)(result)}")
    return result


def calibrate_hk_ts(ts: TimeSeries) -> TimeSeries:
    """Given a housekeeping timeseries, calibrate each column.
    Replaces the values in the columns with calibrated values.
    Columns with names that are not recognized are ignored.
    New

    Parameters
    ----------
    housekeeping_ts
        A timeseries or Table of uncalibrated housekeeping measurements

    Returns
    -------
    calibrated_ts
    """
    cal_ts = ts.copy()
    for this_col in ts.colnames:
        try:
            f = get_calibration_func(this_col)
            cal_ts[this_col] = f(ts[this_col])
        except ValueError:  # ignore columns whose names we do not recognize
            pass
        except KeyError:
            pass
    return cal_ts


def get_calibration_data(hk_name: str) -> Table:
    calibration_data_directory = _data_directory / "calibration"
    data = ascii.read(calibration_data_directory / f"{hk_name}.csv")
    unit_str = data.colnames[-1]
    result = QTable()
    result["adc"] = data["adc"]
    result["value"] = u.Quantity(data[unit_str], unit_str)
    return result
