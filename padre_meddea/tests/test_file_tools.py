from pathlib import Path
import pytest

import numpy as np

import padre_meddea
from padre_meddea.io.file_tools import (
    parse_ph_packets,
    parse_spectrum_packets,
    parse_cmd_response_packets,
    read_file,
)
from padre_meddea.spectrum.spectrum import PhotonList, SpectrumList
from astropy.timeseries import TimeSeries

ph_packet_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
spec_packet_file = padre_meddea._test_files_directory / "apid162_4packets.bin"
hk_packet_file = padre_meddea._test_files_directory / "apid163_4packets.bin"

fits_ph_packet_file = padre_meddea._test_files_directory / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
fits_hk_packet_file = padre_meddea._test_files_directory / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
#fits_spec_packet_file = padre_meddea._test_files_directory / "spec/padre_meddea_l0test_spectrum_20250504T153111_v0.1.0.fits"

NUM_PACKETS = 4


def test_read_file_bad_file():
    """Should return an error if file type is not recognized"""
    with pytest.raises(ValueError):
        read_file(Path("test.jpg"))


def test_read_ph_file():
    pkt_list, ph_list = parse_ph_packets(ph_packet_file)
    # check that there are the right number of events
    # assert len(ph_list) == 760
    # check that there are in fact 4 packets
    assert len(np.unique(pkt_list["seqcount"])) == NUM_PACKETS
    assert len(pkt_list) == NUM_PACKETS


def test_read_spec_file():
    specs = parse_spectrum_packets(spec_packet_file)
    assert len(specs[0]) == NUM_PACKETS


def test_read_file_fits():
    assert isinstance(read_file(fits_ph_packet_file), PhotonList)
    # TODO re-enable this once the test spec files are fixed.
    #assert isinstance(read_file(fits_spec_packet_file), SpectrumList)

    # note that there are no command packets in this file
    hk_ts, cmd_ts = read_file(fits_hk_packet_file)
    assert isinstance(hk_ts, TimeSeries)
    assert isinstance(cmd_ts, TimeSeries)
