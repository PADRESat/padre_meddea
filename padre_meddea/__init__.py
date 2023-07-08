# see license/LICENSE.rst
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

log.debug(f"padre_meddea version: {__version__}")

MISSION_NAME = "PADRE"
INSTRUMENT_NAME = "MEDDEA"
