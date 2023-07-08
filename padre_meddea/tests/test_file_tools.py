from padre_meddea.io.file_tools import read_file


def test_read_file():
    assert read_file("test_file.cdf") is None
