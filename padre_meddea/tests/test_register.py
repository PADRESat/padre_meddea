import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.timeseries import TimeSeries

import padre_meddea.housekeeping.register as register

register_table = register.register_table
address_list = list(register_table["address"])


@pytest.mark.parametrize("address", address_list)
def test_shift_asic_register(address):
    for asic_num in range(4):
        assert (
            register.unshift_asic_reg_addr(
                asic_num, register.shift_asic_reg_addr(asic_num, address)
            )
            == address
        )


def test_load_register_table():
    reg_table = register.load_register_table()
    assert isinstance(reg_table, Table)
    assert (
        len(reg_table) > 200
    )  # let's not be too specific in case there are changes needed to the register_table.csv


def test_add_register_address_name():
    ts = TimeSeries(
        time_start="2016-03-22T12:30:31",
        time_delta=3 * u.s,
        n_samples=len(register_table),
    )
    ts["address"] = address_list
    new_ts = register.add_register_address_name(ts)
    assert len(new_ts) == len(ts)
    assert "name" in new_ts.colnames
    for this_row in new_ts:
        this_name = this_row["name"]
        correct_name = register_table.loc["address", this_row["address"]]["name"]
        if isinstance(
            correct_name, str
        ):  # not sure why this would ever return anything else? are there cases of multiple matches?!
            assert this_name == correct_name


def test_register_table():
    """Ensure that there are no issue with the register table data."""

    # make sure there are no duplicates
    assert len(np.unique(register_table["address"])) == len(register_table)
    assert len(np.unique(register_table["name"])) == len(register_table)

    for this_row in register_table:
        assert (this_row["address"] % 2) == 0  # all register addresses are even
        assert 0 <= this_row["address"] < 0xFFFF  # all address are less than 2 bytes
        assert int(this_row["address_hex"], 16) == this_row["address"]
