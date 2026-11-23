import sqlite3
from contextlib import closing
from hashlib import sha256

import astropy.units as u
import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time
from astropy.timeseries import TimeSeries

from padre_meddea import _test_files_directory
from padre_meddea.db import (
    connect,
    load_housekeeping,
    load_photons,
    read_housekeeping,
    read_photons,
)
from padre_meddea.io import read_file
from padre_meddea.spectrum.spectrum import PhotonList
from padre_meddea.util.util import calc_time


def test_initialize_and_reopen(tmp_path):
    path = tmp_path / "meddea.sqlite"
    with closing(connect(path)) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        connection.execute(
            "INSERT INTO files (id, filename, sha256, data_type) "
            "VALUES (1, 'photon.fits', 'checksum', 'photon')"
        )
        connection.execute(
            "INSERT INTO photon_packets (file_id, row_index, seqcount) VALUES (1, 0, 3)"
        )
        connection.executemany(
            "INSERT INTO photon_events "
            "(file_id, row_index, packet_row_index, seqcount, clocks) "
            "VALUES (1, ?, 0, 3, 10)",
            [(0,), (1,)],
        )
        connection.commit()

    with closing(connect(path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM photon_packets").fetchone() == (
            1,
        )
        assert connection.execute("SELECT COUNT(*) FROM photon_events").fetchone() == (
            2,
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO photon_events (file_id, row_index, packet_row_index) "
                "VALUES (2, 0, 0)"
            )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO photon_events (file_id, row_index, packet_row_index) "
                "VALUES (1, 0, 0)"
            )


@pytest.mark.parametrize(
    ("table", "data_type", "extra_columns", "extra_values"),
    [
        ("photon_packets", "photon", "", ()),
        ("photon_events", "photon", ", packet_row_index", (0,)),
        ("housekeeping", "housekeeping", "", ()),
        ("command_responses", "housekeeping", "", ()),
    ],
)
def test_data_rows_require_an_existing_source(
    table, data_type, extra_columns, extra_values
):
    with closing(connect(":memory:")) as connection:
        connection.execute(
            "INSERT INTO files (id, filename, sha256, data_type) VALUES (1, ?, ?, ?)",
            ("source.fits", "checksum", data_type),
        )
        if table == "photon_events":
            connection.execute(
                "INSERT INTO photon_packets (file_id, row_index) VALUES (1, 0)"
            )
        placeholders = ", ".join("?" for _ in extra_values)
        if placeholders:
            placeholders = ", " + placeholders
        sql = (
            f"INSERT INTO {table} (file_id, row_index{extra_columns}) "
            f"VALUES (?, 0{placeholders})"
        )
        for invalid_id in (None, 2):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(sql, (invalid_id, *extra_values))
        connection.execute(sql, (1, *extra_values))
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute("DELETE FROM files WHERE id = 1")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(f"UPDATE {table} SET file_id = 2")


@pytest.mark.parametrize(
    ("filename", "extensions"),
    [
        (
            "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
            {"SCI": "photon_events", "PKT": "photon_packets"},
        ),
        (
            "padre_meddea_l0_photon_20260704T204013_v1.0.0.fits",
            {"SCI": "photon_events", "PKT": "photon_packets"},
        ),
        (
            "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits",
            {"HK": "housekeeping", "READ": "command_responses"},
        ),
    ],
)
def test_schema_covers_sample_fits_columns(filename, extensions):
    with (
        closing(connect(":memory:")) as connection,
        fits.open(_test_files_directory / filename) as hdus,
    ):
        for extension, table in extensions.items():
            columns = {
                row[1].lower()
                for row in connection.execute(f"PRAGMA table_info({table})")
            }
            fits_columns = {name.lower() for name in hdus[extension].columns.names}
            if "time" in fits_columns:
                fits_columns.remove("time")
                fits_columns.update({"time_part1", "time_part2"})
            if table == "housekeeping" and "timestamp" in fits_columns:
                fits_columns.remove("timestamp")
                fits_columns.add("raw_timestamp")
            assert fits_columns <= columns


def test_load_l0_housekeeping_fits_into_database(tmp_path):
    filename = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    path = tmp_path / "housekeeping.sqlite"
    hk_fields = (
        "timestamp",
        "fp_temp",
        "dib_temp",
        "hvps_temp",
        "hvps_vsense",
        "hvps_csense",
        "csense_15v",
        "csense_33vd",
        "csense_33va",
        "hit_rate",
        "good_cmd_cnt",
        "error_cnt",
        "heater_pwm_duty_cycle",
        "decimation_rate",
        "error_summary",
        "seqcount",
    )
    db_fields = ("raw_timestamp", *hk_fields[1:])
    columns = ", ".join(db_fields)

    with fits.open(filename) as hdus:
        hk_rows = hdus["HK"].data
        cmd_rows = hdus["READ"].data
        with closing(connect(path)) as connection:
            file_id = load_housekeeping(connection, filename)
            assert connection.in_transaction
            assert load_housekeeping(connection, filename) == file_id
            connection.commit()

        with closing(connect(path)) as connection:
            assert load_housekeeping(connection, filename) == file_id
            saved_hk = connection.execute(
                f"SELECT row_index, timestamp, {columns} FROM housekeeping "
                "ORDER BY row_index"
            ).fetchall()
            assert len(saved_hk) == len(hk_rows) > 0
            assert saved_hk[0][1] == "2025-05-04T05:51:38.000000000Z"
            for index, (saved, original) in enumerate(zip(saved_hk, hk_rows)):
                expected_time = calc_time(int(original["timestamp"])).utc
                expected_time.precision = 9
                assert saved[0] == index
                assert saved[1] == expected_time.isot + "Z"
                assert saved[2:] == tuple(int(original[field]) for field in hk_fields)

            saved_cmd = connection.execute(
                "SELECT row_index, time_ticks, time_part1, time_part2, pkttimes, "
                "pktclock, address, value, seqcount, name "
                "FROM command_responses ORDER BY row_index"
            ).fetchall()
            assert len(saved_cmd) == len(cmd_rows) > 0
            for index, (saved, original) in enumerate(zip(saved_cmd, cmd_rows)):
                seconds = int(original["pkttimes"])
                clock = int(original["pktclock"])
                assert saved == (
                    index,
                    seconds * 20_000_000 + clock,
                    float(original["time"][0]),
                    float(original["time"][1]),
                    seconds,
                    clock,
                    int(original["address"]),
                    int(original["value"]),
                    int(original["seqcount"]),
                    str(original["name"]),
                )
            assert connection.execute("SELECT COUNT(*) FROM files").fetchone() == (1,)
            assert connection.execute(
                "SELECT name FROM source_extensions ORDER BY hdu_index"
            ).fetchall() == [("HK",), ("READ",)]
            assert connection.execute(
                "SELECT filename, sha256, primary_header FROM files WHERE id = ?",
                (file_id,),
            ).fetchone() == (
                filename.name,
                sha256(filename.read_bytes()).hexdigest(),
                hdus[0].header.tostring(),
            )


@pytest.mark.parametrize("outer_transaction", [False, True])
def test_load_housekeeping_rolls_back_only_the_failed_import(
    tmp_path, outer_transaction
):
    filename = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    with closing(connect(tmp_path / "housekeeping.sqlite")) as connection:
        connection.execute(
            "CREATE TRIGGER reject_hk BEFORE INSERT ON housekeeping "
            "WHEN NEW.row_index = 1 BEGIN SELECT RAISE(ABORT, 'rejected'); END"
        )
        connection.execute(
            "INSERT INTO files (filename, sha256, data_type) "
            "VALUES ('other.fits', 'other', 'photon')"
        )
        if not outer_transaction:
            connection.commit()
        with pytest.raises(sqlite3.IntegrityError, match="rejected"):
            load_housekeeping(connection, filename)
        assert connection.in_transaction is outer_transaction
        assert connection.execute("SELECT filename FROM files").fetchall() == [
            ("other.fits",)
        ]
        assert connection.execute("SELECT COUNT(*) FROM housekeeping").fetchone() == (
            0,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM source_extensions"
        ).fetchone() == (0,)
        connection.rollback()


def test_load_housekeeping_rejects_non_housekeeping_file(tmp_path):
    filename = (
        _test_files_directory
        / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
    )
    with closing(connect(tmp_path / "housekeeping.sqlite")) as connection:
        with pytest.raises(ValueError, match="not an L0 or L1 housekeeping"):
            load_housekeeping(connection, filename)
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone() == (0,)


def test_load_housekeeping_with_pkttimes_and_empty_read(tmp_path):
    source = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    filename = tmp_path / "housekeeping.fits"
    with fits.open(source) as hdus:
        hdus["HK"].columns.change_name("timestamp", "pkttimes")
        hdus["READ"].data = hdus["READ"].data[:0]
        hdus.writeto(filename)

    with closing(connect(":memory:")) as connection:
        load_housekeeping(connection, filename)
        assert connection.execute(
            "SELECT timestamp, pkttimes, raw_timestamp FROM housekeeping "
            "ORDER BY row_index LIMIT 1"
        ).fetchone() == ("2025-05-04T05:51:38.000000000Z", 799653103, None)
        assert connection.execute(
            "SELECT COUNT(*) FROM command_responses"
        ).fetchone() == (0,)
        connection.rollback()
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone() == (0,)


@pytest.mark.parametrize("unsupported", ["column", "extension"])
def test_load_housekeeping_rejects_unsupported_data(tmp_path, unsupported):
    source = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    filename = tmp_path / "unknown-column.fits"
    with fits.open(source) as hdus:
        if unsupported == "column":
            extra = fits.Column(
                name="unexpected", format="J", array=[0] * len(hdus["HK"].data)
            )
            hdus[1] = fits.BinTableHDU.from_columns(
                hdus["HK"].columns + fits.ColDefs([extra]), header=hdus["HK"].header
            )
        else:
            hdus.append(
                fits.BinTableHDU.from_columns(hdus["READ"].columns, name="EXTRA")
            )
        hdus.writeto(filename)

    with closing(connect(":memory:")) as connection:
        with pytest.raises(ValueError, match="Unsupported"):
            load_housekeeping(connection, filename)
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone() == (0,)


def test_read_housekeeping_matches_fits_reader(tmp_path):
    filename = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    expected, _ = read_file(filename)
    with closing(connect(tmp_path / "housekeeping.sqlite")) as connection:
        load_housekeeping(connection, filename)
        actual = read_housekeeping(connection, expected.time[2], expected.time[6])
        binned = read_housekeeping(
            connection,
            expected.time[2],
            expected.time[6],
            num_points=1,
            aggregate="avg",
        )

    expected = expected[2:7]
    assert isinstance(actual, TimeSeries)
    assert actual.colnames == expected.colnames
    assert np.all(np.abs((actual.time - expected.time).to_value(u.ns)) < 1)
    assert actual.meta == expected.meta
    for column in expected.colnames[1:]:
        assert actual[column].dtype == expected[column].dtype
        assert actual[column].unit == expected[column].unit
        np.testing.assert_array_equal(actual[column], expected[column])
    assert len(binned) == 1
    assert binned["fp_temp"][0] == pytest.approx(np.mean(expected["fp_temp"]))
    assert binned["raw_timestamp"][0] == pytest.approx(np.mean(expected["timestamp"]))


def test_read_housekeeping_boundaries_and_empty_results():
    filename = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    time = Time("2025-05-04T05:51:38", scale="utc")
    with closing(connect(":memory:")) as connection:
        load_housekeeping(connection, filename)
        assert len(read_housekeeping(connection, time, time)) == 1
        assert len(read_housekeeping(connection, time.tai, time.tai)) == 1
        assert (
            len(read_housekeeping(connection, time + 0.1 * u.ns, time + 1 * u.s)) == 0
        )
        assert (
            len(read_housekeeping(connection, time - 1 * u.s, time - 0.1 * u.ns)) == 0
        )
        assert len(read_housekeeping(connection, "2024-01-01", "2024-01-02")) == 0
        with pytest.raises(ValueError, match="time_start"):
            read_housekeeping(connection, "2025-05-05", "2025-05-04")
        with pytest.raises(ValueError, match="scalar"):
            read_housekeeping(connection, Time(["2025-05-04"]), time)
        masked = Time(np.ma.array(2460800.0, mask=True), format="jd", scale="utc")
        with pytest.raises(ValueError, match="masked"):
            read_housekeeping(connection, masked, time)


@pytest.fixture
def binned_housekeeping_database():
    with closing(connect(":memory:")) as connection:
        connection.executemany(
            "INSERT INTO files (id, filename, sha256, data_type) "
            "VALUES (?, ?, ?, 'housekeeping')",
            [(1, "first.fits", "first"), (2, "second.fits", "second")],
        )
        connection.executemany(
            "INSERT INTO housekeeping (file_id, row_index, timestamp, fp_temp, dib_temp) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (1, 0, "2025-01-01T00:00:00.000000000Z", 2, None),
                (1, 1, "2025-01-01T00:00:02.000000000Z", 4, None),
                (1, 2, "2025-01-01T00:00:05.000000000Z", 8, 1),
                (2, 0, "2025-01-01T00:00:05.000000000Z", 12, 3),
                (1, 3, "2025-01-01T00:00:08.000000000Z", 16, 5),
                (1, 4, "2025-01-01T00:00:10.000000000Z", 32, 7),
            ],
        )
        connection.commit()
        yield connection


@pytest.mark.parametrize(
    ("aggregate", "expected"),
    [
        ("avg", [3, 17]),
        ("min", [2, 8]),
        ("max", [4, 32]),
        ("sum", [6, 68]),
        ("count", [2, 4]),
        ("total", [6, 68]),
    ],
)
def test_read_housekeeping_binned_aggregations(
    binned_housekeeping_database, aggregate, expected
):
    result = read_housekeeping(
        binned_housekeeping_database,
        "2025-01-01T00:00:00",
        "2025-01-01T00:00:10",
        num_points=2,
        aggregate=aggregate,
    )
    assert isinstance(result, TimeSeries)
    assert len(result) == 2
    np.testing.assert_array_equal(result["fp_temp"], expected)
    assert result.time.isot.tolist() == [
        "2025-01-01T00:00:02.500",
        "2025-01-01T00:00:07.500",
    ]
    if aggregate not in {"count", "total"}:
        assert result["dib_temp"].mask.tolist() == [True, False]


def test_read_housekeeping_binned_boundaries_and_empty_bins(
    binned_housekeeping_database,
):
    connection = binned_housekeeping_database
    exact = Time("2025-01-01T00:00:05", scale="utc")
    result = read_housekeeping(connection, exact, exact, num_points=5, aggregate="max")
    assert len(result) == 1
    assert result["fp_temp"][0] == 12
    assert result.time[0] == exact
    result = read_housekeeping(
        connection, exact + 0.1 * u.ns, exact + 1 * u.s, num_points=4
    )
    assert len(result) == 0
    result = read_housekeeping(
        connection, exact - 1 * u.s, exact + 1 * u.s, num_points=4, aggregate="sum"
    )
    assert len(result) == 1
    assert result["fp_temp"][0] == 20
    assert not connection.in_transaction


def test_read_housekeeping_binning_preserves_nanosecond_boundaries(
    binned_housekeeping_database,
):
    connection = binned_housekeeping_database
    connection.executemany(
        "INSERT INTO housekeeping (file_id, row_index, timestamp, fp_temp) "
        "VALUES (1, ?, ?, ?)",
        [
            (10, "2025-01-02T00:00:00.000000000Z", 1),
            (11, "2025-01-02T00:00:00.000000001Z", 2),
            (12, "2025-01-02T00:00:00.000000002Z", 4),
        ],
    )
    start = Time("2025-01-02T00:00:00", scale="utc")
    result = read_housekeeping(
        connection,
        start + 0.1 * u.ns,
        start + 1.1 * u.ns,
        num_points=1,
        aggregate="sum",
    )
    assert len(result) == 1
    assert result["fp_temp"][0] == 2
    assert connection.in_transaction
    connection.rollback()


def test_read_housekeeping_binning_preserves_transaction_on_error(
    binned_housekeeping_database,
):
    connection = binned_housekeeping_database
    connection.execute(
        "UPDATE housekeeping SET fp_temp = ? WHERE file_id = 1 AND row_index = 0",
        (2**63 - 1,),
    )
    with pytest.raises(sqlite3.OperationalError, match="integer overflow"):
        read_housekeeping(
            connection,
            "2025-01-01T00:00:00",
            "2025-01-01T00:00:02",
            num_points=1,
            aggregate="sum",
        )
    assert connection.in_transaction
    connection.rollback()
    assert connection.execute(
        "SELECT fp_temp FROM housekeeping WHERE file_id = 1 AND row_index = 0"
    ).fetchone() == (2,)


def test_read_housekeeping_binning_rejects_invalid_options(
    binned_housekeeping_database,
):
    connection = binned_housekeeping_database
    bounds = ("2025-01-01", "2025-01-02")
    for num_points in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="positive integer"):
            read_housekeeping(connection, *bounds, num_points=num_points)
    for aggregate in ("median", "avg); DROP TABLE files; --", None):
        with pytest.raises(ValueError, match="aggregation"):
            read_housekeeping(connection, *bounds, num_points=2, aggregate=aggregate)
    with pytest.raises(ValueError, match="num_points"):
        read_housekeeping(connection, *bounds, aggregate="max")


def test_read_housekeeping_sorts_multiple_sources():
    directory = _test_files_directory / "hk"
    first = directory / "padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    second = directory / "padre_meddea_l0test_housekeeping_20250504T055308_v0.1.0.fits"
    first_hk, _ = read_file(first)
    second_hk, _ = read_file(second)
    with closing(connect(":memory:")) as connection:
        load_housekeeping(connection, second)
        load_housekeeping(connection, first)
        result = read_housekeeping(
            connection, "2025-05-04T05:51:38", "2025-05-04T06:00:00"
        )
    assert len(result) == len(first_hk) + len(second_hk)
    assert np.all(result.time[1:] >= result.time[:-1])
    assert abs((result.time[0] - first_hk.time[0]).to_value(u.ns)) < 1
    assert abs((result.time[-1] - second_hk.time[-1]).to_value(u.ns)) < 1


def test_read_housekeeping_overlapping_sources_with_different_columns(tmp_path):
    source = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    alternative = tmp_path / "alternative.fits"
    with fits.open(source) as hdus:
        expected_counts = {
            "housekeeping": len(hdus["HK"].data),
            "command_responses": len(hdus["READ"].data),
        }
        hdus["HK"].columns.change_name("timestamp", "pkttimes")
        hdus.writeto(alternative)

    with closing(connect(":memory:")) as connection:
        first_id = load_housekeeping(connection, source)
        second_id = load_housekeeping(connection, alternative)
        assert first_id != second_id
        for table, count in expected_counts.items():
            assert count > 0
            assert connection.execute(
                f"SELECT file_id, COUNT(*) FROM {table} GROUP BY file_id ORDER BY file_id"
            ).fetchall() == [(first_id, count), (second_id, count)]
        result = read_housekeeping(
            connection, "2025-05-04T05:51:38", "2025-05-04T05:51:38"
        )
    assert len(result) == 2
    assert result["timestamp"].mask.tolist() == [False, True]
    assert result["pkttimes"].mask.tolist() == [True, False]
    assert [entry["filename"] for entry in result.meta["sources"]] == [
        source.name,
        alternative.name,
    ]


@pytest.mark.parametrize(
    "filename",
    [
        "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits",
        "padre_meddea_l0_photon_20260704T204013_v1.0.0.fits",
    ],
)
def test_photon_fits_round_trip(tmp_path, filename):
    source = _test_files_directory / filename
    expected = read_file(source)
    with closing(connect(tmp_path / "photons.sqlite")) as connection:
        file_id = load_photons(connection, source)
        assert connection.in_transaction
        assert load_photons(connection, source) == file_id
        assert connection.execute(
            "SELECT COUNT(*) FROM photon_events AS e JOIN photon_packets AS p "
            "ON p.file_id = e.file_id AND p.row_index = e.packet_row_index "
            "WHERE e.seqcount = p.seqcount AND e.pkttimes = p.pkttimes "
            "AND e.pktclock = p.pktclock"
        ).fetchone() == (len(expected.event_list),)
        connection.commit()
        result = read_photons(connection, "2024-01-01", "2028-01-01")

    assert isinstance(result, PhotonList)
    assert result.meta == expected.meta
    np.testing.assert_array_equal(
        result.spectrum().flux.value, expected.spectrum().flux.value
    )
    for actual, original in (
        (result.event_list, expected.event_list),
        (result.pkt_list, expected.pkt_list),
    ):
        original = original[np.argsort(original.time, kind="stable")]
        assert actual.colnames == original.colnames
        assert len(actual) == len(original)
        assert actual.meta == original.meta
        assert np.all(np.abs((actual.time - original.time).to_value(u.ns)) < 150)
        for name in original.colnames[1:]:
            assert actual[name].dtype.kind == original[name].dtype.kind
            assert actual[name].dtype.itemsize == original[name].dtype.itemsize
            assert actual[name].unit == original[name].unit
            np.testing.assert_array_equal(actual[name], original[name])


def test_read_photons_inclusive_and_subnanosecond_bounds():
    source = (
        _test_files_directory / "padre_meddea_l0_photon_20260704T204013_v1.0.0.fits"
    )
    with closing(connect(":memory:")) as connection:
        load_photons(connection, source)
        first = Time("2026-07-04T20:40:13.226120550", format="isot", scale="utc")
        at_first = read_photons(connection, first, first)
        assert len(at_first.event_list) > 0
        assert len(at_first.pkt_list) == 1
        assert np.all(at_first.event_list.time == first)
        assert np.all(at_first.pkt_list.time == first)
        after = read_photons(connection, first + 0.1 * u.ns, first + 1 * u.ns)
        assert len(after.event_list) == 0
        assert len(after.pkt_list) == 0
        outside = read_photons(connection, "2024-01-01", "2024-01-02")
        assert isinstance(outside, PhotonList)
        assert len(outside.event_list) == len(outside.pkt_list) == 0
        with pytest.raises(ValueError, match="time_start"):
            read_photons(connection, "2026-07-05", "2026-07-04")
        full = read_photons(connection, "2026-07-04T20:40:00", "2026-07-04T20:45:00")
        later = next(
            time
            for time in full.event_list.time
            if time > first and not np.any(full.pkt_list.time == time)
        )
        events_only = read_photons(connection, later, later)
        assert len(events_only.event_list) > 0
        assert len(events_only.pkt_list) == 0
        assert events_only.pkt_list.colnames == full.pkt_list.colnames


def test_read_photons_keeps_overlapping_sources(tmp_path):
    source = (
        _test_files_directory
        / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
    )
    copy = tmp_path / "other.fits"
    with fits.open(source) as hdus:
        expected_counts = {
            "photon_events": len(hdus["SCI"].data),
            "photon_packets": len(hdus["PKT"].data),
        }
        hdus[0].header["HISTORY"] = "Second source"
        hdus.writeto(copy)
    with closing(connect(":memory:")) as connection:
        first_id = load_photons(connection, source)
        second_id = load_photons(connection, copy)
        assert first_id != second_id
        for table, count in expected_counts.items():
            assert count > 0
            assert connection.execute(
                f"SELECT file_id, COUNT(*) FROM {table} GROUP BY file_id ORDER BY file_id"
            ).fetchall() == [(first_id, count), (second_id, count)]
        result = read_photons(connection, "2025-01-01", "2026-01-01")
    expected = read_file(source)
    assert len(result.event_list) == 2 * len(expected.event_list)
    assert len(result.pkt_list) == 2 * len(expected.pkt_list)
    assert np.all(result.event_list.time[1:] >= result.event_list.time[:-1])
    assert [entry["filename"] for entry in result.meta["sources"]] == [
        source.name,
        copy.name,
    ]


def test_read_photons_with_different_packet_columns():
    old = (
        _test_files_directory
        / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
    )
    new = _test_files_directory / "padre_meddea_l0_photon_20260704T204013_v1.0.0.fits"
    with closing(connect(":memory:")) as connection:
        load_photons(connection, new)
        load_photons(connection, old)
        result = read_photons(connection, "2025-01-01", "2027-01-01")
    count_old = len(read_file(old).pkt_list)
    assert result.pkt_list["pktlength"].mask[:count_old].all()
    assert not result.pkt_list["pktlength"].mask[count_old:].any()
    assert np.all(result.pkt_list.time[1:] >= result.pkt_list.time[:-1])


def test_load_photons_preserves_50ns_clock_steps(tmp_path):
    source = (
        _test_files_directory / "padre_meddea_l0_photon_20260704T204013_v1.0.0.fits"
    )
    filename = tmp_path / "clock-steps.fits"
    with fits.open(source) as hdus:
        rows = hdus["SCI"].data
        rows["seqcount"][:2] = rows["seqcount"][0]
        rows["pkttimes"][:2] = rows["pkttimes"][0]
        rows["pktclock"][:2] = [4522411, 4522412]
        rows["clocks"][:2] = 0
        original = hdus["PKT"]
        expanded = fits.BinTableHDU.from_columns(
            original.columns, nrows=len(original.data) + 1, header=original.header
        )
        for column in original.columns.names:
            expanded.data[column][-1] = original.data[column][0]
        expanded.data["pktclock"][-1] = 4522412
        hdus[2] = expanded
        hdus.writeto(filename)

    with closing(connect(":memory:")) as connection:
        load_photons(connection, filename)
        rows = connection.execute(
            "SELECT timestamp FROM photon_events WHERE row_index IN (0, 1) "
            "ORDER BY row_index"
        ).fetchall()
    assert rows == [
        ("2026-07-04T20:40:13.226120550Z",),
        ("2026-07-04T20:40:13.226120600Z",),
    ]


def test_load_photons_across_batch_boundary(tmp_path):
    source = (
        _test_files_directory
        / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
    )
    filename = tmp_path / "large.fits"
    with fits.open(source) as hdus:
        packet = hdus["PKT"].data[0]
        seqcount = int(packet["seqcount"])
        pkttimes = int(packet["pkttimes"])
        pktclock = int(packet["pktclock"])
        hdus[1] = fits.BinTableHDU.from_columns(
            hdus["SCI"].columns, nrows=32769, header=hdus["SCI"].header, name="SCI"
        )
        hdus["SCI"].data["seqcount"][:] = seqcount
        hdus["SCI"].data["pkttimes"][:] = pkttimes
        hdus["SCI"].data["pktclock"][:] = pktclock
        hdus["SCI"].data["clocks"][:] = 0
        hdus.writeto(filename)
    with closing(connect(":memory:")) as connection:
        load_photons(connection, filename)
        assert connection.execute(
            "SELECT COUNT(*), MIN(row_index), MAX(row_index) FROM photon_events"
        ).fetchone() == (32769, 0, 32768)


def test_load_photons_rolls_back_packets_when_events_fail():
    source = (
        _test_files_directory
        / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
    )
    with closing(connect(":memory:")) as connection:
        connection.execute(
            "CREATE TRIGGER reject_event BEFORE INSERT ON photon_events "
            "WHEN NEW.row_index = 1 BEGIN SELECT RAISE(ABORT, 'rejected'); END"
        )
        with pytest.raises(sqlite3.IntegrityError, match="rejected"):
            load_photons(connection, source)
        assert not connection.in_transaction
        for table in (
            "files",
            "photon_events",
            "photon_packets",
            "source_extensions",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (
                0,
            )


def test_load_photons_rejects_housekeeping_file():
    source = (
        _test_files_directory
        / "hk/padre_meddea_l0test_housekeeping_20250504T055138_v0.1.0.fits"
    )
    with closing(connect(":memory:")) as connection:
        with pytest.raises(ValueError, match="not an L0 or L1 photon"):
            load_photons(connection, source)
        assert connection.execute("SELECT COUNT(*) FROM files").fetchone() == (0,)


def test_photon_event_packet_foreign_key():
    with closing(connect(":memory:")) as connection:
        connection.executemany(
            "INSERT INTO files (id, filename, sha256, data_type) "
            "VALUES (?, ?, ?, 'photon')",
            [(1, "first.fits", "first"), (2, "second.fits", "second")],
        )
        connection.execute(
            "INSERT INTO photon_packets (file_id, row_index) VALUES (2, 0)"
        )
        for file_id, packet_index in ((1, 0), (2, 1), (2, None)):
            with pytest.raises(sqlite3.IntegrityError):
                connection.execute(
                    "INSERT INTO photon_events (file_id, row_index, packet_row_index) "
                    "VALUES (?, 0, ?)",
                    (file_id, packet_index),
                )
        connection.execute(
            "INSERT INTO photon_events (file_id, row_index, packet_row_index) "
            "VALUES (2, 0, 0)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM photon_packets WHERE file_id = 2 AND row_index = 0"
            )


@pytest.mark.parametrize("problem", ["missing", "ambiguous"])
def test_load_photons_rejects_unmatched_or_ambiguous_events(tmp_path, problem):
    if problem == "missing":
        source = (
            _test_files_directory
            / "eventlist/padre_meddea_l0test_photon_20250504T083234_v0.1.0.fits"
        )
    else:
        original_path = (
            _test_files_directory
            / "eventlist/padre_meddea_l0test_photon_20250504T055311_v0.1.0.fits"
        )
        source = tmp_path / "ambiguous.fits"
        with fits.open(original_path) as hdus:
            packet = hdus["PKT"]
            event = hdus["SCI"].data[0]
            keys = ("seqcount", "pkttimes", "pktclock")
            matched = next(
                row for row in packet.data if all(row[k] == event[k] for k in keys)
            )
            expanded = fits.BinTableHDU.from_columns(
                packet.columns, nrows=len(packet.data) + 1, header=packet.header
            )
            for name in packet.columns.names:
                expanded.data[name][: len(packet.data)] = packet.data[name]
                expanded.data[name][-1] = matched[name]
            hdus[2] = expanded
            hdus.writeto(source)
    with closing(connect(":memory:")) as connection:
        expected_error = (
            "no matching packet" if problem == "missing" else "ambiguous packets"
        )
        with pytest.raises(ValueError, match=expected_error):
            load_photons(connection, source)
        for table in (
            "files",
            "photon_events",
            "photon_packets",
            "source_extensions",
        ):
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone() == (
                0,
            )


def test_timestamps_are_utc_text_and_sort_at_photon_resolution():
    with closing(connect(":memory:")) as connection:
        connection.execute(
            "INSERT INTO files (id, filename, sha256, data_type) "
            "VALUES (1, 'source.fits', 'checksum', 'photon')"
        )
        connection.execute(
            "INSERT INTO files (id, filename, sha256, data_type) "
            "VALUES (2, 'hk.fits', 'hk-checksum', 'housekeeping')"
        )
        connection.execute(
            "INSERT INTO photon_packets (file_id, row_index) VALUES (1, 0)"
        )
        earlier = "2026-07-04T20:40:13.123456700Z"
        later = "2026-07-04T20:40:13.123456750Z"
        connection.executemany(
            "INSERT INTO photon_events (file_id, row_index, packet_row_index, timestamp) "
            "VALUES (1, ?, 0, ?)",
            [(0, later), (1, earlier)],
        )
        connection.execute(
            "INSERT INTO housekeeping (file_id, row_index, timestamp, raw_timestamp) "
            "VALUES (2, 0, ?, 1234)",
            (earlier,),
        )
        assert connection.execute(
            "SELECT timestamp FROM photon_events ORDER BY timestamp"
        ).fetchall() == [(earlier,), (later,)]
        assert connection.execute(
            "SELECT timestamp FROM photon_events WHERE timestamp >= ? AND timestamp < ?",
            (earlier, later),
        ).fetchall() == [(earlier,)]
        assert connection.execute(
            "SELECT timestamp, typeof(timestamp), raw_timestamp FROM housekeeping"
        ).fetchone() == (earlier, "text", 1234)
        for table, file_id in (("photon_events", 1), ("housekeeping", 2)):
            for invalid in (1234, "abcd-07-04T20:40:13.123456750Z"):
                with pytest.raises(sqlite3.IntegrityError):
                    if table == "photon_events":
                        connection.execute(
                            "INSERT INTO photon_events "
                            "(file_id, row_index, packet_row_index, timestamp) "
                            "VALUES (?, 2, 0, ?)",
                            (file_id, invalid),
                        )
                    else:
                        connection.execute(
                            "INSERT INTO housekeeping (file_id, row_index, timestamp) "
                            "VALUES (?, 2, ?)",
                            (file_id, invalid),
                        )


def test_reject_unrelated_database(tmp_path):
    path = tmp_path / "other.sqlite"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute("CREATE TABLE unrelated (value TEXT)")
        connection.commit()

    with pytest.raises(ValueError, match="non-empty"):
        connect(path)

    with closing(sqlite3.connect(path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE name = 'unrelated'"
        ).fetchone() == ("unrelated",)


@pytest.mark.parametrize("version", [1, 2, 3, 5])
def test_reject_unsupported_version(tmp_path, version):
    path = tmp_path / "meddea.sqlite"
    with closing(connect(path)) as connection:
        connection.execute(f"PRAGMA user_version = {version}")

    with pytest.raises(ValueError, match=f"schema version={version}"):
        connect(path)
