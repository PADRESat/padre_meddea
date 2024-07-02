from pathlib import Path
import pytest

from padre_meddea.io.file_tools import read_file


def test_read_file_bad_file():
    """Should return an error if file type is not recognized"""
    with pytest.raises(ValueError):
        read_file(Path('test.jpg'))