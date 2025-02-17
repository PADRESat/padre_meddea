"""Module to provide calibration functions for housekeeping data"""
import numpy as np
from scipy.optimize import fsolve

from astropy.io import ascii
import astropy.units as u

from padre_meddea import _package_directory

_calibdata_directory = _package_directory / "data" / "calibration" / "data"
cal_table = ascii.read(_package_directory / "data" / "calibration" / "padre_meddea_calib_hk_0.csv")
cal_table.add_index("name")


def get_hk_cal_func(hk_name: str):
    """Given a housekeeping name, return the calibration function to convert from
    uncalibrated data values to physical values."""
    if hk_name in list(cal_table["name"]):  # is it a housekeeping
        row = cal_table.loc[hk_name]
        params = [float(p) for p in row["cal_params"].split(",")]
        if row["cal_func"] == "poly":
            f = np.poly1d(params)
        if row["cal_func"] == "log":
            f = lambda x: params[0] * np.log(x) + params[1]
    else:
        raise ValueError(f"The housekeeping name, {hk_name}, is not recognized.")
    return f


def get_hk_cal_inv_func(hk_name: str, x: float) -> float:
    """Given a housekeeping name, return the inverse function to convert from physical values to
    uncalibrated data values."""
    f = get_hk_cal_func(hk_name)
    # TODO: add tests and use calibration data to find closer point
    # find closest point using calibration data
    # from padre_meddea_gse import _data_directory
    # calib_directory = _data_directory / "calibration"
    # calib_data = ascii.read(calib_directory / "hvps_temp.csv")
    return fsolve(f - x, 0.0)


def calibrate_hk_value(hk_name: str, value: int):
    """Given a housekeeping name and a measurement, return the calibrated value."""
    f = get_hk_cal_func(hk_name)
    return f(value)
