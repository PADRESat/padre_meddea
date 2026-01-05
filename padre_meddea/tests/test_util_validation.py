from pathlib import Path
import padre_meddea
from padre_meddea.util import validation


def test_validate_packet_checksums():
    test_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
    warnings = validation.validate_packet_checksums(test_file)
    assert len(warnings) == 0


def test_validate_file_size_within_limit(tmp_path, monkeypatch):
    """Test that a file within size limits produces no warnings."""
    # Create a small test file
    test_file = tmp_path / "small_file.bin"
    test_file.write_bytes(b"small content")

    warnings = validation.validate_file_size(test_file, size_limit=1024)
    assert len(warnings) == 0


def test_validate_file_size_exceeds_limit(tmp_path, monkeypatch):
    """Test that a file exceeding size limits produces a warning."""
    # Create a test file
    test_file = tmp_path / "large_file.bin"
    test_file.write_bytes(b"content")

    # Create a simple mock stat result
    class MockStat:
        st_size = 1024 * 1024 * 100  # 100 MB

    # Save original stat method
    original_stat = Path.stat

    # Define wrapper that returns mock for our test file
    def mock_stat(self):
        if self == test_file:
            return MockStat()
        return original_stat(self)

    # Patch at the class level
    monkeypatch.setattr(Path, "stat", mock_stat)

    warnings = validation.validate_file_size(test_file, size_limit=1024)
    assert len(warnings) == 1
    assert "FileSizeWarning" in warnings[0]
    assert "exceeds expected size" in warnings[0]


def test_validate():
    test_file = padre_meddea._test_files_directory / "apid160_4packets.bin"
    warnings = validation.validate(test_file)
    assert len(warnings) == 0
