from padre_meddea import _test_files_directory
from padre_meddea.spectrum import spectrum
from padre_meddea.io.file_tools import read_raw_a0, read_raw_a2, read_raw_u8

f1 = _test_files_directory / 'apid160_4packets.bin'
f2 = _test_files_directory / 'apid162_4packets.bin'
f3 = _test_files_directory / 'apid163_4packets.bin'


def test_spectrumlist():
    """Test that we can create a spectrumlist from a raw file"""
    speclist = read_raw_a2(f2)
    assert isinstance(speclist, spectrum.SpectrumList)


def test_photonlist():
    """Test that we can create a spectrumlist from a raw file"""
    phlist = read_raw_a0(f1)
    assert isinstance(phlist, spectrum.PhotonList)