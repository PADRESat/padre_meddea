from pathlib import Path

import boto3
import pytest
from astropy.io import fits
from astropy.time import Time
from moto import mock_aws

import padre_meddea
import padre_meddea.calibration.calibration as calib
import padre_meddea.io.aws_db as aws_db
import padre_meddea.io.file_tools as file_tools


@pytest.fixture(scope="function")
def aws_credentials(monkeypatch):
    """Mocked AWS credentials for moto."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_SECURITY_TOKEN", "testing")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="function")
def mocked_timestream(aws_credentials):
    """Create a mocked Timestream database and table for tests."""
    with mock_aws():
        client = boto3.client("timestream-write", region_name="us-east-1")
        client.create_database(DatabaseName="dev-padre_sdc_aws_logs")
        client.create_table(
            DatabaseName="dev-padre_sdc_aws_logs",
            TableName="dev-padre_measures_table",
            RetentionProperties={
                "MemoryStoreRetentionPeriodInHours": 24,
                "MagneticStoreRetentionPeriodInDays": 7,
            },
        )
        yield client


@pytest.mark.parametrize(
    "bin_file,expected_data_type,expect_aws_upload",
    [
        ("padreMDA0_260704194521_cal.dat", "photon", True),
        ("padreMDA0_260704194521_flare.dat", "photon", True),
        ("padreMDA0_260704194521_particles.dat", "photon", True),
        ("padreMDA0_240916122901.dat", "photon", False),
        ("padreMDA2_240916122851.dat", "spectrum", True),
        ("padreMDU8_240916122904.dat", "housekeeping", True),
    ],
)
def test_process_file_test_files(
    bin_file,
    expected_data_type,
    expect_aws_upload,
    tmpdir,
    monkeypatch,
    mocked_timestream,
):
    """Test processing different file types with output saved to a temporary directory"""
    # Set up the temporary directory as the current working directory
    monkeypatch.chdir(tmpdir)

    recorded = []

    def fake_record_timeseries(ts, table, mission):
        recorded.append((ts, table, mission))

    monkeypatch.setattr(aws_db, "record_timeseries", fake_record_timeseries)
    monkeypatch.setattr(aws_db, "create_annotation", lambda *args, **kwargs: None)

    # Process the File
    files = calib.process_file(
        padre_meddea._test_files_directory / bin_file, overwrite=False
    )
    assert Path(files[0]).exists()
    with fits.open(files[0]) as f:
        assert f[0].header["INSTRUME"] == "MeDDEA"

    # Check that the filename includes the correct data type
    assert f"padre_meddea_l0_{expected_data_type}_" in files[0].name
    assert files[0].name.endswith(".fits")
    if expect_aws_upload:
        assert [table for _, table, _ in recorded].count("files_raw") == 1
        assert [table for _, table, _ in recorded].count("files_l0") == 1
        assert all(mission == "meddea" for _, _, mission in recorded)

        data_table = (
            "housekeeping" if expected_data_type == "housekeeping" else "spectra"
        )
        data_records = [ts for ts, table, _ in recorded if table == data_table]
        assert len(data_records) == 1
        assert len(data_records[0]) > 0

    match expected_data_type:
        case "photon":
            photon_list = file_tools.read_fits_l0l1_photon(files[0])
            pkt_list, event_list = photon_list.pkt_list, photon_list.event_list
            assert all(pkt_list.time > Time("2024-01-01T00:00"))
            assert all(event_list.time > Time("2024-01-01T00:00"))
        case "spectrum":
            spectrum_list = file_tools.read_fits_l0l1_spectrum(files[0])
            assert all(spectrum_list.time > Time("2024-01-01T00:00"))
        case "housekeeping":
            hk_ts, cmd_ts = file_tools.read_fits_l0l1_housekeeping(files[0])
            assert all(hk_ts.time > Time("2024-01-01T00:00"))
            if "time" in cmd_ts.colnames:  # Check if command response times are present
                # If command response times are present, check them
                assert all(cmd_ts.time > Time("2024-01-01T00:00"))
