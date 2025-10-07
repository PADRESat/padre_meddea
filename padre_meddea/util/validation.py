"""
This module contains utilities for file and packet validation.
"""

from typing import List

import numpy as np
from ccsdspy import utils

from padre_meddea import APID, log
from padre_meddea.util.util import is_consecutive


def validate_packet_checksums(file, output_bad_packets: bool = False) -> List[str]:
    """
    Custom Validation Function to check that all packets have contents that match their checksums. This is achieved be a rolling XOR of the packet contents. If the final XOR value is not 0, a warning is issued.

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.

    output_bad_packets: bool
        If true than provide bad packets in return

    Returns
    -------
    List of strings, each in the format "WarningType: message", describing potential validation issues. Returns an empty list if no warnings are issued.
    """
    validation_warnings = []
    bad_packets = []
    # Read the file
    packets = utils.split_packet_bytes(file)
    for i, packet in enumerate(packets):
        # Convert to an array of u16 integers
        packet_arr = np.frombuffer(packet, dtype=np.uint8)
        checksum_validation = np.bitwise_xor.reduce(packet_arr)
        # Make sure the XOR reduction is 0
        if checksum_validation != 0:
            validation_warnings.append(
                f"ChecksumWarning: Packet number {i} has a checksum error."
            )
            bad_packets.append(packet)
    if len(validation_warnings) == 0:
        log.info("No checksum errors found.")
        bad_packets = None
    else:
        percent_error = len(validation_warnings) / len(packets)
        log.info(
            f"ChecksumWarning: Found {len(validation_warnings)}/{len(packets)} checksum errors ({percent_error * 100:0.2f}%)"
        )
    if output_bad_packets:
        return validation_warnings, bad_packets
    else:
        return validation_warnings


def validate_ccsds_apids(file, output_bad_packets: bool = False) -> List[str]:
    """
    Custom Validation Function to check that all packets have contents that match their checksums. This is achieved be a rolling XOR of the packet contents. If the final XOR value is not 0, a warning is issued.

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.

    output_bad_packets: bool
        If true than provide bad packets in return

    Returns
    -------
    List of strings, each in the format "WarningType: message", describing potential validation issues. Returns an empty list if no warnings are issued.
    """
    validation_warnings = []
    bad_packets = []
    # Read the file
    stream_by_apid = utils.split_by_apid(file)
    for this_apid, these_packets in stream_by_apid.items():
        if this_apid not in APID.values():
            for this_packet in these_packets:
                validation_warnings.append(
                    f"ChecksumWarning: Found unexpected APID {this_apid}"
                )
                bad_packets.append(this_packet)
    if output_bad_packets:
        return validation_warnings, bad_packets
    else:
        return validation_warnings


def validate_ccsds_seqnum(file) -> List[str]:
    """
    Custom Validation Function to check that all packets have contents that match their checksums. This is achieved be a rolling XOR of the packet contents. If the final XOR value is not 0, a warning is issued.

    Parameters
    ----------
    file: `str | BytesIO`
        A file path (str) or file-like object with a `.read()` method.

    output_bad_packets: bool
        If true than provide bad packets in return

    Returns
    -------
    List of strings, each in the format "WarningType: message", describing potential validation issues. Returns an empty list if no warnings are issued.
    """
    validation_warnings = []
    header_arrays = utils.read_primary_headers(file)
    sequence_count = header_arrays["CCSDS_SEQUENCE_COUNT"]
    min_seq = sequence_count.min()
    max_seq = sequence_count.max()
    if is_consecutive(sequence_count):
        log.info(f"No missing packets from sequence count {min_seq} to {max_seq}")
    else:
        missing_seqs = sorted(set(range(min_seq, max_seq)) - set(sequence_count))
        for this_missed_seq in missing_seqs:
            validation_warnings.append(
                f"PacketMissWarning: Packet number {this_missed_seq} is missing."
            )
            percent_error = len(validation_warnings) / len(sequence_count)
            log.info(
                f"ChecksumWarning: Found {len(validation_warnings)}/{len(sequence_count)} checksum errors ({percent_error * 100:0.2f}%)"
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
    ccsdspy_warnings = utils.validate(file, APID.values())
    validation_warnings.extend(ccsdspy_warnings)
    # Run custom validation functions
    if custom_validators:
        for validator in custom_validators:
            # Execute Custom Validator
            custom_warnings = validator(file)
            validation_warnings.extend(custom_warnings)

    return validation_warnings
