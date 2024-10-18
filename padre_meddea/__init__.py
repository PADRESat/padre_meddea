# see license/LICENSE.rst
from pathlib import Path

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from swxsoc import config as swxsoc_config, log as swxsoc_log, print_config
from astropy.time import Time

# Load user configuration
config = swxsoc_config

log = swxsoc_log

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["config", "print_config"]

_package_directory = Path(__file__).parent
_data_directory = _package_directory / "data"
_test_files_directory = _package_directory / "data" / "test"

MISSION_NAME = "PADRE"
INSTRUMENT_NAME = "MeDDEA"

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

APID = {
    "spectrum": 0xA2,  # decimal 162
    "photon": 0xA0,  # decimal 160
    "housekeeping": 0xA3,  # decimal 163
    "cmd_resp": 0xA5,  # decimal 165
}

EPOCH = Time("2000-01-01 00:00", scale="utc")

log.debug(f"padre_meddea version: {__version__}")
