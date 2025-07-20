import padre_meddea
from padre_meddea.util import validation


def test_validate_packet_checksums():
    test_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
    warnings = validation.validate_packet_checksums(test_file)
    assert len(warnings) == 0


def test_validate():
    test_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
    warnings = validation.validate(test_file)
    assert len(warnings) == 0
