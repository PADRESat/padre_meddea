"""Provides functions to calibrate housekeeping data"""

import astropy.units as u
import numpy as np
from astropy.io import ascii
from astropy.table import QTable, Table
from astropy.timeseries import TimeSeries
from scipy.optimize import fsolve

from padre_meddea.util.util import calc_time

from .housekeeping import _data_directory

calibration_table = ascii.read(_data_directory / "padre_meddea_calib_hk_0.csv")
calibration_table.add_index("name")

# define the conversion from MEDDEA housekeeping names to SpaceOps housekeeping names
MEDDEA_TO_SPACEOPS_HK_CONVERSION = {
    "hvps_temp": "HVTemp",
    "hvps_csense": "HVCurrent",
    "error_summary": "sysError",
    "ph_rate": "phRate",
    "good_cmd_count": "goodCmdCount",
    "time_stamp1": "TimeStamp1",
    "csense_33vd": "Amps_3V3_D",
    "time_stamp2": "TimeStamp2",
    "csense_33va": "Amps_3V3_A",
    "sequence_count": "SequenceCount",
    "dib_temp": "DIBTemp",
    "csense_15v": "Amps_1V5",
    "decimation_rate": "decimationRate",
    "hvps_vsense": "HVVolts",
    "fp_temp": "FPTemp",
    "error_cnt": "errorCount",
    "heater_pwm_duty_cycle": "heaterPWM",
}


def calibrate_spaceops_hk_values(hk_dict: dict) -> dict:
    """Given a dictionary of housekeeping values from SpaceOps, calibrate the values and return a new dictionary with the same keys but calibrated values.

    Parameters
    ----------
    hk_dict : dict
        A dictionary of uncalibrated housekeeping values from SpaceOps. The keys should be the same as those in SPACEOPS_HK_CONVERSION.

    Returns
    -------
    calibrated_hk_dict : dict
        A new dictionary with the same keys as hk_dict but with calibrated values.
    """
    # Swap keys and values
    SPACEOPS_TO_MEDDEA_HK_CONVERSION = {
        v: k for k, v in MEDDEA_TO_SPACEOPS_HK_CONVERSION.items()
    }
    calibrated_hk_dict = {}
    for key, value in hk_dict.items():
        if key in SPACEOPS_TO_MEDDEA_HK_CONVERSION:
            try:
                calib_func = get_calibration_func(SPACEOPS_TO_MEDDEA_HK_CONVERSION[key])
                calibrated_hk_dict[key] = calib_func(value)
            except ValueError:
                calibrated_hk_dict[key] = value
        else:
            calibrated_hk_dict[key] = value
    timestamp = np.array(
        [hk_dict["TimeStamp2"], hk_dict["TimeStamp1"]], dtype="uint16"
    ).view(dtype="uint32")
    calibrated_hk_dict["time"] = calc_time(timestamp)
    return calibrated_hk_dict


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
    """
    Create a B-spline interpolation function for converting ADC values to physical values.
    This function retrieves calibration data for a specified housekeeping parameter
    and fits a B-spline interpolation to the ADC-to-value mapping. The returned
    function can be used to convert raw ADC readings to calibrated physical values
    with appropriate units.

    Parameters
    ----------
    hk_name : str
        Name of the housekeeping parameter for which to create the calibration function.

    Returns
    -------
    callable
        A function that takes ADC values as input and returns the corresponding
        calibrated physical values as a `u.Quantity` object with appropriate units.

    Notes
    -----
    The returned function handles unit conversion automatically based on the units
    of the calibration data.
    """

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
    """
    Retrieve calibration data for a specific housekeeping parameter.
    This function reads calibration data from a CSV file named after the housekeeping
    parameter and returns it as an astropy QTable with ADC values and corresponding
    physical values with appropriate units.

    Parameters
    ----------
    hk_name : str
        Name of the housekeeping parameter for which to retrieve calibration data.
        This will be used to find the corresponding CSV file.
    Returns
    -------
    astropy.table.Table
        A table containing the calibration data with columns:
        - 'adc': ADC values (raw digital counts)
        - 'value': Physical values with appropriate units

    Notes
    -----
    The calibration CSV file is expected to have at least two columns:
    one named 'adc' and another column whose name represents the physical unit.
    """
    calibration_data_directory = _data_directory / "calibration"
    data = ascii.read(calibration_data_directory / f"{hk_name}.csv")
    unit_str = data.colnames[-1]
    result = QTable()
    result["adc"] = data["adc"]
    result["value"] = u.Quantity(data[unit_str], unit_str)
    return result


def parse_error_summary(hk_ts: TimeSeries) -> TimeSeries:
    """Given a time series with an error summary column, parse it out to its individual elements.

    Parameters
    ----------
    hk_ts : TimeSeries

    Returns
    -------
    error_summary_ts : TimeSeries
    """
    if "error_summary" not in hk_ts.colnames:
        raise ValueError("Missing error_summary column.")
    error_ts = TimeSeries(
        time=hk_ts.time, data={"error_summary": hk_ts["error_summary"]}
    )
    col_names = [
        "uart_parity_err",
        "ping1_parity_err",
        "ping2_parity_err",
        "hist_parity_err",
        "heater_err",
        "caliste_reg_err",
        "caliste_seu",
        "rogue_pps",
        "fake_evt_flg",
        "freewheel_pps",
    ]
    for i, this_col in enumerate(col_names):
        error_ts[this_col] = (error_ts["error_summary"] & 2**i) > 0
    return error_ts
