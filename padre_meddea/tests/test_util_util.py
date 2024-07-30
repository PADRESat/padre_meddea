"""Tests for util.py"""

import pytest

from astropy.time import Time
from padre_meddea.util import util

time = "2024-04-06T12:06:21"
time_formatted = "20240406T120621"


# fmt: off
@pytest.mark.parametrize("instrument,time,level,descriptor,version,result", [
    ("meddea", time, "l1", "eventlist", "1.2.3", f"padre_meddea_l1_eventlist_{time_formatted}_v1.2.3.fits"),
    ("meddea", time, "l2", "spec", "2.4.5", f"padre_meddea_l2_spec_{time_formatted}_v2.4.5.fits"),
    ("meddea", time, "l2", "spec-eventlist", "1.3.5", f"padre_meddea_l2_spec-eventlist_{time_formatted}_v1.3.5.fits"),
    ("meddea", time, "l3", "xraydirect", "2.4.5", f"padre_meddea_l3_xraydirect_{time_formatted}_v2.4.5.fits"),
]
)
def test_science_filename_output_a(instrument, time, level, descriptor, version, result):
    """Test simple cases with expected output"""
    assert (
        util.create_science_filename(instrument, time=time, level=level, descriptor=descriptor, version=version)
        == result
    )
# fmt: on


def test_science_filename_output_b():
    """Test more complex cases of expected output"""

    # mode
    assert (
        util.create_science_filename(
            "meddea", time, descriptor="spec", level="l3", mode="2s", version="2.4.5"
        )
        == f"padre_meddea_2s_l3_spec_{time_formatted}_v2.4.5.fits"
    )
    # test
    assert (
        util.create_science_filename(
            "meddea",
            time,
            descriptor="eventlist",
            level="l1",
            version="2.4.5",
            test=True,
        )
        == f"padre_meddea_l1test_eventlist_{time_formatted}_v2.4.5.fits"
    )
    # all options
    assert (
        util.create_science_filename(
            "meddea",
            time,
            level="l3",
            mode="2s",
            descriptor="spec",
            version="2.4.5",
            test=True,
        )
        == f"padre_meddea_2s_l3test_spec_{time_formatted}_v2.4.5.fits"
    )
    # Time object instead of str
    assert (
        util.create_science_filename(
            "meddea",
            Time(time),
            level="l3",
            mode="2s",
            descriptor="spec",
            version="2.4.5",
            test=True,
        )
        == f"padre_meddea_2s_l3test_spec_{time_formatted}_v2.4.5.fits"
    )
    # Time object but created differently
    assert (
        util.create_science_filename(
            "meddea",
            Time(2460407.004409722, format="jd"),
            level="l3",
            mode="2s",
            descriptor="eventlist",
            version="2.4.5",
            test=True,
        )
        == f"padre_meddea_2s_l3test_eventlist_{time_formatted}_v2.4.5.fits"
    )


def test_parse_science_filename_output():
    """Test for known outputs"""
    # all parameters
    input = {
        "instrument": "meddea",
        "mode": "2s",
        "level": "l3",
        "test": False,
        "descriptor": "spec",
        "version": "2.4.5",
        "time": Time("2024-04-06T12:06:21"),
    }

    f = util.create_science_filename(
        input["instrument"],
        input["time"],
        input["level"],
        input["version"],
        test=input["test"],
        descriptor=input["descriptor"],
        mode=input["mode"],
    )
    assert util.parse_science_filename(f) == input

    # test only
    input = {
        "instrument": "meddea",
        "level": "l3",
        "test": True,
        "version": "2.4.5",
        "time": Time("2024-04-06T12:06:21"),
        "mode": None,
        "descriptor": "spec",
    }

    f = util.create_science_filename(
        input["instrument"],
        time=input["time"],
        level=input["level"],
        version=input["version"],
        descriptor=input["descriptor"],
        test=input["test"],
    )
    assert util.parse_science_filename(f) == input

    # descriptor only
    input = {
        "instrument": "meddea",
        "mode": None,
        "level": "l3",
        "test": False,
        "descriptor": "eventlist",
        "version": "2.4.5",
        "time": Time("2024-04-06T12:06:21"),
    }

    f = util.create_science_filename(
        input["instrument"],
        input["time"],
        input["level"],
        input["version"],
        descriptor=input["descriptor"],
    )
    assert util.parse_science_filename(f) == input

    # mode only
    input = {
        "instrument": "meddea",
        "mode": "2s",
        "level": "l2",
        "test": False,
        "descriptor": "eventlist",
        "version": "2.7.9",
        "time": Time("2024-04-06T12:06:21"),
    }

    f = util.create_science_filename(
        input["instrument"],
        time=input["time"],
        level=input["level"],
        version=input["version"],
        descriptor=input["descriptor"],
        mode=input["mode"],
    )
    assert util.parse_science_filename(f) == input


def test_parse_science_filename_errors_l1():
    """Test for errors in l1 and above files"""
    with pytest.raises(ValueError):
        # wrong mission name
        f = "veeger_spn_2s_l3test_burst_20240406_120621_v2.4.5"
        util.parse_science_filename(f)

        # wrong instrument name
        f = "hermes_www_2s_l3test_burst_20240406_120621_v2.4.5"
        util.parse_science_filename(f)


good_time = "2025-06-02T12:04:01"
good_instrument = "meddea"
good_level = "l1"
good_version = "1.3.4"


# fmt: off
@pytest.mark.parametrize(
    "instrument,time,level,version",
    [
        (good_instrument, good_time, good_level, "1.3"),  # bad version specifications
        (good_instrument, good_time, good_level, "1"),
        (good_instrument, good_time, good_level, "1.5.6.7"),
        (good_instrument, good_time, good_level, "1.."),
        (good_instrument, good_time, good_level, "a.5.6"),
        (good_instrument, good_time, "la", good_version),  # wrong level specifications
        (good_instrument, good_time, "squirrel", good_version),
        ("potato", good_time, good_level, good_version),  # wrong instrument names
        ("eeb", good_time, good_level, good_version),
        ("fpi", good_time, good_level, good_version),
        (good_instrument, "2023-13-04T12:06:21", good_level, good_version),  # non-existent time
        (good_instrument, "2023/13/04 12:06:21", good_level, good_version),  # not isot format
        (good_instrument, "2023/13/04 12:06:21", good_level, good_version),  # not isot format
        (good_instrument, "12345345", good_level, good_version),  # not valid input for time
    ]
)
def test_science_filename_errors_l1_a(instrument, time, level, version):
    """"""
    with pytest.raises(ValueError) as e:
        util.create_science_filename(
            instrument, time, descriptor="eventlist", level=level, version=version
        )
# fmt: on


def test_science_filename_errors_l1_b():
    with pytest.raises(ValueError):
        # _ character in mode
        util.create_science_filename(
            "eeb",
            time="12345345",
            descriptor="eventlist",
            level=good_level,
            version=good_version,
            mode="o_o",
        )
    with pytest.raises(ValueError):
        # _ character in descriptor
        util.create_science_filename(
            "eeb",
            time="12345345",
            level=good_level,
            version=good_version,
            descriptor="blue_green",
        )
