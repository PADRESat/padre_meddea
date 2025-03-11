"""
This module contains utilities for file and packet validation.
"""

from typing import List

import numpy as np
from ccsdspy import utils


def validate_packet_checksums(file) -> List[str]:
    """
    Custom Validation Function to check that all packets have contents that match their checksums. This is achieved be a rolling XOR of the packet contents. If the final XOR value is not 0, a warning is issued.

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.

    Returns
    -------
    List of strings, each in the format "WarningType: message", describing potential validation issues. Returns an empty list if no warnings are issued.
    """
    validation_warnings = []
    # Read the file
    packets = utils.split_packet_bytes(file)
    for i, packet in enumerate(packets):
        # Convert to an array of u16 integers
        packet_arr = np.frombuffer(packet, dtype=np.uint8)
        checksum_validation = np.bitwise_xor.reduce(packet_arr)
        # Make sure the XOR reduction is 0
        if checksum_validation != 0:
            validation_warnings.append(
                f"ChecksumWarning: Packet {i} has a checksum error."
            )
    return validation_warnings


def validate(
    file, valid_apids: List[int] = None, custom_validators: List[callable] = None
) -> List[str]:
    """
    Validate a file containing CCSDS packets and capturing any exceptions or warnings they generate.
    This function checks:

    - Primary header consistency (sequence counts in order, no missing sequence numbers, found APIDs)
    - File integrity (truncation, extra bytes)

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.
    valid_apids: `list[int]| None`, optional
       Optional list of valid APIDs. If specified, warning will be issued when
       an APID is encountered outside this list.
    custom_validators: `List[callable]`, optional
        List of custom validation functions that take a file-like object as input and return a list of warnings

    Returns
    -------
    List of strings, each in the format "WarningType: message", describing
    potential validation issues. Returns an empty list if no warnings are issued.
    """
    validation_warnings = []
    # Run Baseline CCSDSPy validation
    ccsdspy_warnings = utils.validate(file, valid_apids)
    validation_warnings.extend(ccsdspy_warnings)
    # Run custom validation functions
    if custom_validators:
        for validator in custom_validators:
            # Execute Custom Validator
            custom_warnings = validator(file)
            validation_warnings.extend(custom_warnings)

    return validation_warnings
