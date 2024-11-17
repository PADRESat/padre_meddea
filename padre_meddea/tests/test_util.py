import pytest

import numpy as np

import padre_meddea
from padre_meddea import EPOCH
import padre_meddea.util.util as util

from astropy.time import TimeDelta
import astropy.units as u

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


def test_is_not_consecutive():
    assert not util.is_consecutive([0, 2, 3, 4, 5])
    assert not util.is_consecutive(
        np.concatenate((np.arange(1, 10), np.arange(11, 20)))
    )


@pytest.mark.parametrize(
    "input,output",
    [
        (26, 0),
        (15, 1),
        (8, 2),
        (1, 3),
        (29, 4),
        (13, 5),
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
def test_channel_to_pix(input, output):
    assert util.channel_to_pixel(input) == output


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
