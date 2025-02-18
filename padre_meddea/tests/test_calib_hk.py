import pytest

import numpy as np

import astropy.units as u
from astropy.io import ascii

from padre_meddea.calibration import calib_hk as calib

cal_table = calib.cal_table

calib_directory = calib._calibdata_directory

hk_names = list(calib.cal_table["name"])


@pytest.mark.parametrize("name", hk_names)
def test_get_calib_func(name):
    """Test that we can get calibration functions for all hk names"""
    # check that it returns a function
    f = calib.get_calib_func(name)
    assert callable(f)
    # check that it takes an integer and returns a float
    assert isinstance(f(1), u.Quantity)  # cannot test 0 because log(0)


@pytest.mark.parametrize("name", hk_names)
def test_get_hk_cal_func(name):
    """Test that we can get calibration functions for all hk names"""
    # check that it returns a function
    f = calib.get_hk_cal_func(name)
    assert callable(f)
    # check that it takes an integer and returns a float
    assert isinstance(f(1), float)  # cannot test 0 because log(0)


@pytest.mark.parametrize("name", ["error_cnt", "good_cmd_cnt", "blah", "hit_rate"])
def test_get_hk_cal_func_error(name):
    """Test that we get none if the name is not known"""
    # check that it returns a vallueError
    with pytest.raises(ValueError):
        calib.get_hk_cal_func(name)


@pytest.mark.parametrize("name", hk_names)
def test_hk_min_values(name):
    """Test that the min values in adc have the correct min values in physical units"""
    # check that it returns a function
    f = calib.get_hk_cal_func(name)
    row = cal_table.loc[name]


#    assert np.isclose(f(row["low_limit"]), row["low_limit_phys"])
#    assert np.isclose(f(row["high_limit"]), row["high_limit_phys"])


@pytest.mark.parametrize("name", hk_names)
def test_cal_table_units(name):
    """Test that all units are interpretable by astropy units"""
    row = cal_table.loc[name]
    assert isinstance(u.Quantity(1, row["phys_unit"]), u.Quantity)


def test_hvps_temp_calibration():
    calib_data = ascii.read(calib_directory / "hvps_temp.csv")
    f = calib.get_hk_cal_func("hvps_temp")
    assert np.allclose(
        np.array(calib_data["deg_C"]), f(np.array(calib_data["adc"])), rtol=0.2
    )


def test_hvps_setpoint_calibration():
    calib_data = ascii.read(calib_directory / "hvps_setpoint.csv")
    f = calib.get_hk_cal_func("hvps_setpoint")
    assert np.allclose(
        np.array(calib_data["volt"]), f(np.array(calib_data["adc"])), rtol=0.2
    )

    # test the inverse transformation
    f_inv = calib.get_hk_cal_func("hvps_setpoint_inv")
    assert np.allclose(
        f_inv(np.array(calib_data["volt"])), np.array(calib_data["adc"]), rtol=0.2
    )


def test_hvps_vsense_calibration():
    calib_data = ascii.read(calib_directory / "hvps_vsense.csv")
    f = calib.get_hk_cal_func("hvps_vsense")
    assert np.allclose(
        np.array(calib_data["volt"]), f(np.array(calib_data["adc"])), rtol=2
    )


def test_pulser_setpoint_calibration():
    calib_data = ascii.read(calib_directory / "pulser_setpoint.csv")
    f = calib.get_hk_cal_func("pulser_setpoint")
    assert np.allclose(
        np.array(calib_data["volt"]), f(np.array(calib_data["adc"])), rtol=0.2
    )


def test_fp_temp_calibration():
    calib_data = ascii.read(calib_directory / "fp_temp.csv")
    f = calib.get_hk_cal_func("fp_temp")
    # only valid over a limited range
    y = np.array(calib_data["deg_C"])[6:]
    x = np.array(calib_data["adc"])[6:]
    assert np.allclose(y, f(x), rtol=1)


def test_heater_setpoint_calibration():
    calib_data = ascii.read(calib_directory / "fp_temp.csv")
    f = calib.get_hk_cal_func("heater_setpoint")
    y = np.array(calib_data["deg_C"])[6:]
    x = np.array(calib_data["adc"])[6:]
    assert np.allclose(y, f(x), rtol=1)


@pytest.mark.parametrize("name", ["csense_15v", "csense_33vd", "csense_33va"])
def test_current_sense_calibration(name):
    f = calib.get_hk_cal_func(name)
    adc = [20000, 30000, 40000]
    mamps = [230.65, 122.89, 15.13]
    for this_adc, this_current in zip(adc, mamps):
        assert np.allclose([float(f(this_adc))], [this_current], 0.2)


def test_hvps_temp_calibration():
    calib_data = ascii.read(calib_directory / "hvps_temp.csv")
    f = calib.get_hk_cal_func("hvps_temp")
    assert np.allclose(
        np.array(calib_data["deg_C"]), f(np.array(calib_data["adc"])), rtol=2
    )


def test_dib_temp_calibration():
    calib_data = ascii.read(calib_directory / "dib_temp.csv")
    f = calib.get_hk_cal_func("dib_temp")
    assert np.allclose(
        np.array(calib_data["deg_C"]), f(np.array(calib_data["adc"])), rtol=2
    )
