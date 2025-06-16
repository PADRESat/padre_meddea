import pytest

import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose
from padre_meddea.util import pulser


@pytest.mark.parametrize(
    "test_input,expected",
    [
        (0, 1000 * u.Hz),
        (1, 233.81968 * u.Hz),
        (2, 132.38721 * u.Hz),
        (10, 29.613836 * u.Hz),
    ],
)
def test_pulser_frequency_output(test_input, expected):
    assert_quantity_allclose([pulser.pulser_frequency(test_input)], [expected])
