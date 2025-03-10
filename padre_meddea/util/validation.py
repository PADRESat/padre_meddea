"""
This module contains utilities for file and packet validation. 
"""

import numpy as np
from ccsdspy.utils import validate


def is_consecutive(arr: np.array) -> bool:
    """Return True if the packet sequence numbers are all consecutive integers, has no missing numbers."""
    MAX_SEQCOUNT = 2**14 - 1  # 16383

    # Ensure arr is at least 1D
    arr = np.atleast_1d(arr)

    # check if seqcount has wrapped around
    indices = np.where(arr == MAX_SEQCOUNT)
    if len(indices[0]) == 0:  # no wrap
        return np.all(np.diff(arr) == 1)
    else:
        last_index = 0
        result = True
        for this_ind in indices[0]:
            this_arr = arr[last_index : this_ind + 1]
            result = result & np.all(np.diff(this_arr) == 1)
            last_index = this_ind + 1
        # now do the remaining part of the array
        this_arr = arr[last_index + 1 :]
        result = result & np.all(np.diff(this_arr) == 1)
        return result


def validate(file, valid_apids=None):
    """
    Validate a file containing CCSDS packets and capturing any exceptions or warnings they generate.
    This function checks:

    - Primary header consistency (sequence counts in order, no missing sequence numbers, found APIDs)
    - File integrity (truncation, extra bytes)

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.
    """
    return validate(file, valid_apids)
