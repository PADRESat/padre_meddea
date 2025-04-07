# see license/LICENSE.rst
import os
from pathlib import Path

import numpy as np

from astropy.time import Time
import astropy.io
from astropy.table import vstack

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

# Get SWXSOC_MISSIONS environment variable if it exists or use default for mission
SWXSOC_MISSION = os.getenv("SWXSOC_MISSION", "padre")
os.environ["SWXSOC_MISSION"] = SWXSOC_MISSION

from swxsoc import (  # noqa: E402
    config as swxsoc_config,
    log as swxsoc_log,
    print_config,
)

# Load user configuration
config = swxsoc_config

log = swxsoc_log

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["config", "print_config"]

_package_directory = Path(__file__).parent
_data_directory = _package_directory / "data"
_test_files_directory = _package_directory / "data" / "test"

register_table = astropy.io.ascii.read(
    _data_directory / "register_table.csv",
    converters={"address_hex": str},
    format="csv",
)
register_table.add_index("name")
register_table["address"] = [
    int(addr, 16) for addr in list(register_table["address_hex"])
]
register_table.add_index("address")
asic_ind = (register_table["address"] >= 0x0020) * (register_table["address"] <= 0x0044)


def shift_asic_reg_addr(asic_num: int, addr: int):
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


def unshift_asic_reg_addr(asic_num: int, addr: int):
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

# the ratio of detector area for large pixels versus small pixels
RATIO_TOTAL_LARGE_TO_SMALL_PIX = 0.947

# the ratio of a large pixel to a small pixel area
RATIO_LARGE_TO_SMALL_PIX = 9.47

NUM_PIXELS = 12
NUM_SMALL_PIXELS = 4
NUM_LARGE_PIXELS = NUM_PIXELS - NUM_SMALL_PIXELS

peaking_time = [
    0.73,
    1.39,
    2.05,
    2.72,
    3.39,
    4.06,
    4.72,
    5.39,
    6.07,
    6.73,
    7.40,
    8.04,
    8.73,
    9.39,
    10.06,
    10.73,
]

# the bin edges for the histogram data product
HIST_BINS = np.arange(0, 4097, 8, dtype=np.uint16)

APID = {
    "spectrum": 0xA2,  # decimal 162
    "photon": 0xA0,  # decimal 160
    "housekeeping": 0xA3,  # decimal 163
    "cmd_resp": 0x99,  # decimal 153
}

EPOCH = Time("2000-01-01 00:00", scale="utc")

log.debug(f"padre_meddea version: {__version__}")
