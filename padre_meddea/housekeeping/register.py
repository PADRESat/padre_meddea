"""Provides tools to parse register read housekeeping data"""

from astropy.io import ascii
from astropy.table import Table
from astropy.timeseries import TimeSeries

from padre_meddea import _data_directory, log


def shift_asic_reg_addr(asic_num: int, addr: int) -> int:
    """Shift a raw asic register address to the correct address.

    Parameters
    ----------
    asic_num : int
        The asic number
    addr : int
        The raw asic register address
    Return
    ------
    int
      The shifted register address
    """
    base = 0x40 * asic_num
    return base + addr


def unshift_asic_reg_addr(asic_num: int, addr: int) -> int:
    """Unshift a shifted asic register address to the raw address.

    Parameters
    ----------
    asic_num : int
        The asic number
    addr : int
        The shifted asic register address
    Return
    ------
    int
      The unshifted asic register address
    """
    base = 0x40 * asic_num
    return addr - base


def load_register_table() -> Table:
    """Load the register table and add the asic registers.

    Parameters
    ----------
    None

    Returns
    -------
    Table
        register_table
    """
    register_table = ascii.read(
        _data_directory / "register_table.csv",
        converters={"address_hex": str},
        format="csv",
    )
    register_table.add_index("name")
    register_table.add_index("address")
    register_table.sort("address")
    register_table_reordered = register_table[
        ["name", "address", "address_hex", "description"]
    ]
    return register_table_reordered


register_table = load_register_table()


def add_register_address_name(ts: TimeSeries) -> TimeSeries:
    """Given a command time series, add a new column with the name of the register as a string."""
    name_list = [""] * len(ts)
    for i, this_row in enumerate(ts):
        try:
            row = register_table.loc["address", this_row["address"]]
            name_list[i] = row["name"]
        except KeyError:
            log.warning(
                f"Found unknown address in READ timeseries, {this_row['address']}"
            )
            name_list[i] = "unknown"
    ts["name"] = [str(this_name) for this_name in name_list]
    return ts
