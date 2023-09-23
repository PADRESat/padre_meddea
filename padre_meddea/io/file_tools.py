"""
This module provides a generic file reader.
"""
import ccsdspy
from ccsdspy.utils import split_packet_bytes
from ccsdspy.utils import split_by_apid
from ccsdspy import PacketField, PacketArray

__all__ = ["read_file"]

APID_HIST = 0xA2  #  162
APID_PHOTON = 0xA0  #  160


def read_file(data_filename):
    """
    Read a file.

    Parameters
    ----------
    data_filename: str
        A file to read.

    Returns
    -------
    data: str

    Examples
    --------
    """
    result = read_l0_file(data_filename)
    return result


def read_l0_file(data_filename: str, include_ccsds_headers: bool = True):
    """
    Read a level 0 data file.

    Parameters
    ----------
    data_filename : str
        A file to read.
    include_ccsds_headers : bool
        If True then return the CCSDS headers in the data arrays.

    Returns
    -------
    data : dict
        A dictionary of data arrays. Keys are the APIDs.
    """
    with open(data_filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    result = {}
    # TODO move this message to the logger
    print(f"APIds found {stream_by_apid.keys()}.")

    if APID_HIST in stream_by_apid.keys():
        packet_def = packet_definition_hist()
        pkt = ccsdspy.FixedLength(packet_def)
        data = pkt.load(
            stream_by_apid[APID_HIST], include_primary_header=include_ccsds_headers
        )
        result.update({APID_HIST: data})
    elif APID_PHOTON in stream_by_apid.keys():
        packet_def = packet_definition_ph()
        pkt = ccsdspy.VariableLength(packet_def)
        result = pkt.load(
            stream_by_apid[APID_PHOTON], include_primary_header=include_ccsds_headers
        )
        result.update({APID_PHOTON: data})

    return result


def packet_definition_hist():
    """Return the packet definition for the histogram packets."""
    # the number of pixels provided by a histogram packet
    NUM_PIXELS = 48

    p = [
        PacketField(name="START_TIME", data_type="uint", bit_length=4 * 16),
        PacketField(name="END_TIME", data_type="uint", bit_length=4 * 16),
    ]

    for i in range(NUM_PIXELS):
        p += [
            PacketField(name=f"HISTOGRAM_SYNC{i}", data_type="uint", bit_length=8),
            PacketField(name=f"HISTOGRAM_DETNUM{i}", data_type="uint", bit_length=3),
            PacketField(name=f"HISTOGRAM_PIXNUM{i}", data_type="uint", bit_length=5),
            PacketArray(
                name=f"HISTOGRAM_DATA{i}",
                data_type="uint",
                bit_length=16,
                array_shape=256,
            ),
        ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_ph():
    """Return the packet definition for the photon packets."""
    p = [
        PacketField(name="TIME", data_type="uint", bit_length=4 * 16),
        PacketField(name="INT_TIME", data_type="uint", bit_length=16),
        PacketField(name="FLAGS", data_type="uint", bit_length=16),
        PacketField(name="CHECKSUM", data_type="uint", bit_length=16),
        PacketArray(
            name="PIXEL_DATA", data_type="uint", bit_length=16, array_shape="expand"
        ),
    ]
    return p
