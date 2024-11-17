# see license/LICENSE.rst
import os
from pathlib import Path

from astropy.time import Time

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
config["mission"]["mission_name"] = "padre"
config["mission"]["file_extension"] = ".fits"
config["mission"]["inst_names"] = ["meddea", "sharp"]
config["mission"]["inst_fullnames"] = [
    "Measuring Directivity to Determine Electron Anisotropy",
    "sharp",
]
config["mission"]["inst_to_shortname"] = {"meddea": "meddea", "sharp": "sharp"}


log = swxsoc_log

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["config", "print_config"]

_package_directory = Path(__file__).parent
_data_directory = _package_directory / "data"
_test_files_directory = _package_directory / "data" / "test"

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
    "cmd_resp": 0x99,  # decimal 153
}

EPOCH = Time("2000-01-01 00:00", scale="utc")

log.debug(f"padre_meddea version: {__version__}")
