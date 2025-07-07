import pytest

import numpy as np

import padre_meddea
from padre_meddea import EPOCH
import padre_meddea.util.util as util

from astropy.time import TimeDelta
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose

TIME = "2024-04-06T12:06:21"
TIME_FORMATTED = "20240406T120621"


# fmt: off
@pytest.mark.parametrize("instrument,time,level,version,result", [
    ("meddea", TIME, "l1", "1.2.3", f"padre_meddea_l1_{TIME_FORMATTED}_v1.2.3.fits"),
    ("meddea", TIME, "l2", "2.4.5", f"padre_meddea_l2_{TIME_FORMATTED}_v2.4.5.fits"),
    ("sharp", TIME, "l2", "1.3.5", f"padre_sharp_l2_{TIME_FORMATTED}_v1.3.5.fits"),
    ("sharp", TIME, "l3", "2.4.5", f"padre_sharp_l3_{TIME_FORMATTED}_v2.4.5.fits"),
]
)
def test_science_filename_output_a(instrument, time, level, version, result):
    """Test simple cases with expected output.
    Since we are using the swxsoc create_science_filename, we are testing whether we did the config correctly in __init__.py"""
    assert (
        util.create_science_filename(instrument, time, level=level, version=version)
        == result
    )
# fmt: on


def test_is_consecutive():
    """Test if consecutive"""
    assert util.is_consecutive(np.arange(10))
    assert util.is_consecutive(range(100))
    assert util.is_consecutive(np.arange(10, 100, 1))
    # test if the array loops over
    assert util.is_consecutive(
        np.concatenate((np.arange(0, 2**14), np.arange(0, 2000)))
    )


def test_is_not_consecutive():
    """Test if not consecutive"""
    assert not util.is_consecutive([0, 2, 3, 4, 5])
    assert not util.is_consecutive(
        np.concatenate((np.arange(1, 10), np.arange(11, 20)))
    )


@pytest.mark.parametrize(
    "channel,pixel_num",
    [
        (26, 0),
        (15, 1),
        (8, 2),
        (1, 3),
        (29, 4),
        (18, 5),
        (5, 6),
        (0, 7),
        (30, 8),
        (21, 9),
        (11, 10),
        (3, 11),
        (31, 12),
        (14, 14 + 12),  # unconnected channel gets 12 added to it
    ],
)
def test_channel_to_pix(channel, pixel_num):
    assert util.channel_to_pixel(channel) == pixel_num
    if pixel_num < 12:
        assert util.pixel_to_channel(pixel_num) == channel


@pytest.mark.parametrize(
    "pixel_id,asic_num,pixel_num",
    [
        (51738, 0, 0),
        (51720, 0, 2),
        (51730, 0, 5),
        (51712, 0, 7),
        (51733, 0, 9),
        (51715, 0, 11),
        (51770, 1, 0),
        (51752, 1, 2),
        (51762, 1, 5),
        (51744, 1, 7),
        (51765, 1, 9),
        (51747, 1, 11),
        (51802, 2, 0),
        (51784, 2, 2),
        (51794, 2, 5),
        (51776, 2, 7),
        (51797, 2, 9),
        (51779, 2, 11),
        (51834, 3, 0),
        (51816, 3, 2),
        (51826, 3, 5),
        (51808, 3, 7),
        (51829, 3, 9),
        (51811, 3, 11),
    ],
)
def test_get_pixelid(pixel_id, asic_num, pixel_num):
    assert util.get_pixelid(asic_num, pixel_num) == pixel_id
    assert util.parse_pixelids(pixel_id) == (asic_num, util.pixel_to_channel(pixel_num))


def test_has_baseline():
    assert not util.has_baseline(
        padre_meddea._test_files_directory / "apid160_4packets.bin"
    )


def test_has_baseline_error():
    with pytest.raises(ValueError):
        util.has_baseline(padre_meddea._test_files_directory / "apid162_4packets.bin")


@pytest.mark.parametrize(
    "pkt_time_s,pkt_time_clk,ph_clk,output",
    [
        (0, 0, 0, EPOCH),
        (1, 0, 0, EPOCH + TimeDelta(1 * u.s)),
        (10, 0, 0, EPOCH + TimeDelta(10 * u.s)),
        (0, 1, 0, EPOCH + TimeDelta(0.05 * u.microsecond)),
        (0, 0, 1, EPOCH + TimeDelta(12.8 * u.microsecond)),
        (
            5,
            5,
            5,
            EPOCH
            + TimeDelta(5 * u.s + 5 * 0.05 * u.microsecond + 5 * 12.8 * u.microsecond),
        ),
    ],
)
def test_calc_time(pkt_time_s, pkt_time_clk, ph_clk, output):
    assert util.calc_time(pkt_time_s, pkt_time_clk, ph_clk) == output


def test_pixel_to_string():
    assert util.pixel_to_str(0) == "Pixel0L"
    assert util.pixel_to_str(8) == "Pixel8S"
    assert util.pixel_to_str(11) == "Pixel11S"


def test_pixel_to_string_error():
    with pytest.raises(ValueError):
        util.pixel_to_str(13)
    with pytest.raises(ValueError):
        util.pixel_to_str(16)


def test_threshold_to_energy_error():
    with pytest.raises(ValueError):
        util.threshold_to_energy(-1)
    with pytest.raises(ValueError):
        util.threshold_to_energy(64)
    with pytest.raises(ValueError):
        util.threshold_to_energy(100)


@pytest.mark.parametrize(
    "input,output",
    [
        (0, -0.6 * u.keV),
        (55, 10.4 * u.keV),
        (62, 16 * u.keV),
    ],
)
def test_threshold_to_energy(input, output):
    assert_quantity_allclose([util.threshold_to_energy(input)], [output])
