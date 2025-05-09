"""Utilities for read data."""

from astropy.timeseries import TimeSeries
from padre_meddea import register_table


def add_address_name(ts: TimeSeries) -> TimeSeries:
    """Given a command time series, add a new column with the name of the register as a string."""
    name_list = [""] * len(ts)
    for i, this_row in enumerate(ts):
        try:
            row = register_table.loc["address", this_row["address"]]
        except KeyError:
            pass
        if row:
            name_list[i] = row["name"]
    ts["name"] = [str(this_name) for this_name in name_list]
    return ts
