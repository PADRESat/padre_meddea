"""
This module provides tools to analyze and manipulate spectrum data.
"""

import numpy as np

from astropy.timeseries import aggregate_downsample
from specutils import Spectrum1D


def phlist_to_lc(event_list, int_time):
    """Convert an event list to a light curve.

    Parameters
    ----------
    event_list: An event list

    Returns
    -------
    event_list

    Examples
    --------
    """
    ts = aggregate_downsample(event_list, time_bin_size=int_time, aggregate_func=np.sum)
    return ts


def phlist_to_spec(event_list, bins=None):
    if bins is None:
        bins = np.arange(0, 2**12 - 1)
    data, bins = np.histogram(event_list["energy"], bins=bins)
    result = Spectrum1D(
        flux=u.Quantity(data, "count"),
        spectral_axis=u.Quantity(bins, "pix"),
        uncertainty=np.sqrt(data),
    )
    return result
