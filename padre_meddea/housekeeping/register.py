from astropy.io import ascii
from astropy.table import vstack, Table
from astropy.timeseries import TimeSeries

from padre_meddea import _data_directory


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
    register_table["address"] = [
        int(addr, 16) for addr in list(register_table["address_hex"])
    ]

    # find the indices for the asic registers
    asic_ind = (register_table["address"] >= 0x0020) * (register_table["address"] <= 0x0044)

    for this_asic in range(4):
        this_register_asic_table = register_table[asic_ind].copy()
        for this_row in this_register_asic_table:
            this_row["name"] = this_row["name"].replace("asic", f"asic{this_asic}")
            this_row["address"] = shift_asic_reg_addr(this_asic, this_row["address"])
            this_row["address_hex"] = f"{this_row['address']:04x}"
        if this_asic == 0:
            register_asic_table = this_register_asic_table
        else:
            register_asic_table = vstack([register_asic_table, this_register_asic_table])

    register_table = register_table[~asic_ind]
    register_table = vstack([register_table, register_asic_table])
    register_table.add_index("address")
    return register_table

register_table = load_register_table()


def add_register_address_name(ts: TimeSeries) -> TimeSeries:
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