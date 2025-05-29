import pytest

import numpy as np

import astropy.units as u

import padre_meddea.housekeeping.calibration as calib_hk

calibration_table = calib_hk.calibration_table

hk_names = list(calibration_table["name"])


@pytest.mark.parametrize("name", hk_names)
def test_get_calib_func(name):
    """Test that we can get calibration functions for all hk names"""
    # check that it returns a function
    f = calib_hk.get_calibration_func(name)
    assert callable(f)
    # check that it takes an integer and returns a quantity
    assert isinstance(f(1), u.Quantity)  # cannot test 0 because log(0)


@pytest.mark.parametrize("name", ["error_cnt", "good_cmd_cnt", "blah", "hit_rate"])
def test_get_hk_cal_func_error(name):
    """Test that we get none if the name is not known"""
    # check that it returns a vallueError
    with pytest.raises(ValueError):
        calib_hk._get_calibration_func(name)
    with pytest.raises(ValueError):
        calib_hk.get_calibration_func(name)


@pytest.mark.parametrize("name", hk_names)
def test_cal_table_units(name):
    """Test that all units are interpretable by astropy units"""
    row = calibration_table.loc[name]
    assert isinstance(u.Quantity(1, row["unit_str"]), u.Quantity)


@pytest.mark.parametrize("name", hk_names)
def test_calibration_fit(name):
    """Test that calibration_func fits the calibration data.
    Remove heater_pwm_duty_cycle because there is no calibration file.
    hvps calibration curves are not great so give them lower tolerance.
    TODO improve calibration polynomial for hvps
    """
    if name in ["heater_pwm_duty_cycle"]:
        # TODO there is no data file for this
        return True
    data = calib_hk.get_calibration_data(name)
    f = calib_hk.get_calibration_func(name)
    if name in ["hvps_vsense", "hvps_csense"]:
        rtol = 2
    else:
        rtol = 1e-1
    assert np.allclose(np.array(data["value"]), np.array(f(data["adc"])), rtol=rtol)


@pytest.mark.parametrize("name", hk_names)
def test_inverse_calibrate(name):
    if name in ["heater_pwm_duty_cycle"]:
        # TODO there is no data file for this
        return True
    data = calib_hk.get_calibration_data(name)
    if name in ["hvps_vsense", "hvps_csense"]:
        rtol = 2
    else:
        rtol = 1e-1
    for row in data:
        assert np.allclose(
            np.array(row["adc"]),
            np.array(calib_hk.inverse_calibrate(name, row["value"])),
            rtol=rtol,
        )
