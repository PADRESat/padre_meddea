"""
This module provides general utility functions.
"""

__all__ = ["VALID_DATA_LEVELS"]

TIME_FORMAT_L0 = "%Y%j-%H%M%S"
TIME_FORMAT = "%Y%m%dT%H%M%S"
VALID_DESCRIPTORS = ["eventlist", "spec-eventlist", "spec", "xraydirect"]
VALID_DATA_LEVELS = ["l0", "l1", "l2", "l3", "l4"]
FILENAME_EXTENSION = ".fits"
