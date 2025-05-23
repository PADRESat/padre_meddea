import pytest
from padre_meddea import _test_files_directory
from padre_meddea.housekeeping import housekeeping as hk

hk_packet_file = _test_files_directory / "apid163_4packets.bin"

NUM_PACKETS = 4


def test_read_hk_file():
    hk_list = hk.parse_housekeeping_packets(hk_packet_file)
    assert len(hk_list) == NUM_PACKETS


def test_hk_packet_definition():
    packet_definition = hk.packet_definition_hk()
    assert (
        len(packet_definition) == len(hk.hk_definitions) + 2
    )  # add one for checksum and one for timestamp
