# see license/LICENSE.rst
from pathlib import Path

try:
    from ._version import version as __version__
    from ._version import version_tuple
except ImportError:
    __version__ = "unknown version"
    version_tuple = (0, 0, "unknown version")

from padre_meddea.util.config import load_config, print_config
from padre_meddea.util.logger import _init_log

# Load user configuration
config = load_config()

log = _init_log(config=config)

# Then you can be explicit to control what ends up in the namespace,
__all__ = ["config", "print_config"]

_package_directory = Path(__file__).parent
_data_directory = _package_directory / "data"

MISSION_NAME = "PADRE"
INSTRUMENT_NAME = "MEDDEA"

log.debug(f"padre_meddea version: {__version__}")
