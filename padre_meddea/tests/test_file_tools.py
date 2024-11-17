from pathlib import Path
import pytest

import numpy as np

import padre_meddea

from padre_meddea.io.file_tools import (
    parse_ph_packets,
    parse_hk_packets,
    parse_spectrum_packets,
    parse_cmd_response_packets,
    read_file,
)

ph_packet_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
hk_packet_file = padre_meddea._test_files_directory / "apid163_4packets.bin"
spec_packet_file = padre_meddea._test_files_directory / "apid162_4packets.bin"

NUM_PACKETS = 4


def test_read_file_bad_file():
    """Should return an error if file type is not recognized"""
    with pytest.raises(ValueError):
        read_file(Path("test.jpg"))


def test_read_ph_file():
    ph_list, pkt_list = parse_ph_packets(ph_packet_file)
    # check that there are the right number of events
    # assert len(ph_list) == 760
    # check that there are in fact 4 packets
    assert len(np.unique(ph_list["seqcount"])) == NUM_PACKETS
    assert len(pkt_list) == NUM_PACKETS


def test_read_hk_file():
    hk_list = parse_hk_packets(hk_packet_file)
    assert len(hk_list) == NUM_PACKETS


def test_read_spec_file():
    specs = parse_spectrum_packets(spec_packet_file)
    assert len(specs[0]) == NUM_PACKETS
