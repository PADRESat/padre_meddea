"""Load MeDDEA data products into a local SQLite database."""

import math
import re
import sqlite3
from contextlib import contextmanager
from hashlib import sha256
from numbers import Integral, Real
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.time import Time, TimeDelta
from astropy.timeseries import TimeSeries

from padre_meddea import EPOCH
from padre_meddea.spectrum.spectrum import PhotonList
from padre_meddea.util.util import calc_time

__all__ = ["load_housekeeping", "read_housekeeping", "load_photons", "read_photons"]

_TICKS_PER_SECOND = 20_000_000


def _integer(value: object, field: str) -> int:
    if not isinstance(value, Integral):
        raise ValueError(f"FITS column {field!r} must contain scalar integers.")
    return int(value)


def _table_columns(
    hdu: fits.BinTableHDU, allowed: set[str], aliases: dict[str, str]
) -> list[tuple[str, str]]:
    columns: list[tuple[str, str]] = []
    seen: set[str] = set()
    for column in hdu.columns:
        name = column.name.lower()
        target = aliases.get(name, name)
        if (
            not target.isascii()
            or not target.isidentifier()
            or target not in allowed
            or target in seen
        ):
            raise ValueError(
                f"Unsupported or duplicate {hdu.name} column: {column.name}"
            )
        if column.null is not None:
            raise ValueError(
                f"FITS null values are not supported in {hdu.name}.{name}."
            )
        seen.add(target)
        columns.append((column.name, target))
    return columns


@contextmanager
def _import_transaction(connection: sqlite3.Connection):
    started_transaction = not connection.in_transaction
    if started_transaction:
        connection.execute("BEGIN IMMEDIATE")
    connection.execute("SAVEPOINT import_fits")
    try:
        yield
        connection.execute("RELEASE SAVEPOINT import_fits")
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK TO SAVEPOINT import_fits")
            connection.execute("RELEASE SAVEPOINT import_fits")
            if started_transaction:
                connection.rollback()
        raise


def _store_source(
    connection: sqlite3.Connection,
    path: Path,
    digest: str,
    level: str,
    hdus: fits.HDUList,
    data_type: str,
) -> tuple[int, bool]:
    existing = connection.execute(
        "SELECT id, data_type FROM files WHERE sha256 = ?", (digest,)
    ).fetchone()
    if existing:
        if existing[1] != data_type:
            raise ValueError(f"File checksum belongs to a non-{data_type} source.")
        return existing[0], False
    cursor = connection.execute(
        "INSERT INTO files (filename, sha256, data_type, level, primary_header) "
        "VALUES (?, ?, ?, ?, ?)",
        (path.name, digest, data_type, level, hdus[0].header.tostring()),
    )
    file_id = cursor.lastrowid
    if file_id is None:
        raise sqlite3.DatabaseError("Could not retrieve the file ID.")
    connection.executemany(
        "INSERT INTO source_extensions (file_id, hdu_index, name, header) "
        "VALUES (?, ?, ?, ?)",
        (
            (file_id, index, hdu.name, hdu.header.tostring())
            for index, hdu in enumerate(hdus[1:], 1)
        ),
    )
    return file_id, True


def load_housekeeping(connection: sqlite3.Connection, filename: str | Path) -> int:
    """Load an L0 or L1 housekeeping FITS file and return its source ID.

    Identical files are imported only once, based on their SHA-256 digest.
    All original HK and READ fields supported by the schema and the FITS
    headers are retained. Unsupported columns or additional data extensions
    raise an error instead of being discarded.

    The import is atomic and does not commit the caller's transaction.
    Call ``connection.commit()`` to persist it, or ``rollback()`` to discard it.
    The connection must have foreign-key enforcement enabled, as with
    :func:`padre_meddea.db.connect`.
    """
    if not connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise ValueError("Database connection must enable foreign keys.")
    path = Path(filename)
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        with fits.open(stream, memmap=False) as hdus:
            header = hdus[0].header
            data_type = str(header.get("BTYPE", "")).lower()
            level = str(header.get("LEVEL", "")).lower()
            if data_type != "housekeeping" or level not in {"l0", "l1"}:
                raise ValueError(f"{path} is not an L0 or L1 housekeeping FITS file.")

            names = [hdu.name for hdu in hdus[1:]]
            if names.count("HK") != 1 or names.count("READ") != 1:
                raise ValueError(
                    "A housekeeping FITS file must contain one HK "
                    "and one READ extension."
                )
            hk_hdu = hdus["HK"]
            read_hdu = hdus["READ"]
            if not isinstance(hk_hdu, fits.BinTableHDU) or not isinstance(
                read_hdu, fits.BinTableHDU
            ):
                raise ValueError("HK and READ must be FITS binary tables.")
            for hdu in hdus[1:]:
                if (
                    hdu.name not in {"HK", "READ"}
                    and hdu.data is not None
                    and hdu.data.size
                ):
                    raise ValueError(f"Unsupported data extension: {hdu.name}")

            hk_allowed = {
                row[1] for row in connection.execute("PRAGMA table_info(housekeeping)")
            } - {"file_id", "row_index", "timestamp"}
            read_allowed = {
                row[1]
                for row in connection.execute("PRAGMA table_info(command_responses)")
            } - {"file_id", "row_index", "time_ticks", "time_part1", "time_part2"}
            hk_columns = _table_columns(
                hk_hdu, hk_allowed, {"timestamp": "raw_timestamp"}
            )
            read_columns = _table_columns(read_hdu, read_allowed | {"time"}, {})
            hk_names = {target for _, target in hk_columns}
            read_names = {target for _, target in read_columns}
            if not hk_names.intersection({"pkttimes", "raw_timestamp"}):
                raise ValueError("HK must contain pkttimes or timestamp.")
            if len(read_hdu.data) and not {"pkttimes", "pktclock"} <= read_names:
                raise ValueError("READ rows require pkttimes and pktclock.")

            with _import_transaction(connection):
                file_id, is_new = _store_source(
                    connection, path, digest.hexdigest(), level, hdus, "housekeeping"
                )
                if is_new:
                    hk_targets = [target for _, target in hk_columns]
                    hk_sql_columns = ", ".join(f'"{name}"' for name in hk_targets)
                    hk_placeholders = ", ".join("?" for _ in hk_targets)
                    time_target = (
                        "pkttimes" if "pkttimes" in hk_names else "raw_timestamp"
                    )
                    hk_time_field = next(
                        name for name, target in hk_columns if target == time_target
                    )
                    for index, row in enumerate(hk_hdu.data):
                        timestamp = calc_time(
                            _integer(row[hk_time_field], hk_time_field)
                        ).utc
                        timestamp.precision = 9
                        values = [_integer(row[name], name) for name, _ in hk_columns]
                        connection.execute(
                            "INSERT INTO housekeeping "
                            f"(file_id, row_index, timestamp, {hk_sql_columns}) "
                            f"VALUES (?, ?, ?, {hk_placeholders})",
                            (file_id, index, timestamp.isot + "Z", *values),
                        )

                    read_fields = [
                        (name, target)
                        for name, target in read_columns
                        if target != "time"
                    ]
                    read_targets = [target for _, target in read_fields]
                    read_sql_columns = ", ".join(f'"{name}"' for name in read_targets)
                    read_placeholders = ", ".join("?" for _ in read_targets)
                    read_time_name = next(
                        (name for name, target in read_columns if target == "time"),
                        None,
                    )
                    read_source_names = {target: name for name, target in read_fields}
                    for index, row in enumerate(read_hdu.data):
                        seconds = _integer(
                            row[read_source_names["pkttimes"]], "pkttimes"
                        )
                        clock = _integer(row[read_source_names["pktclock"]], "pktclock")
                        time_parts: tuple[float | None, float | None] = (None, None)
                        if read_time_name is not None:
                            raw_time = row[read_time_name]
                            if (
                                not isinstance(raw_time, np.ndarray)
                                or raw_time.shape != (2,)
                                or not all(
                                    math.isfinite(float(value)) for value in raw_time
                                )
                            ):
                                raise ValueError(
                                    "READ time must contain two finite values."
                                )
                            time_parts = (float(raw_time[0]), float(raw_time[1]))
                        values: list[int | str] = []
                        for name, target in read_fields:
                            value = row[name]
                            if target == "name":
                                if not isinstance(value, str):
                                    raise ValueError("READ name must contain text.")
                                values.append(value)
                            else:
                                values.append(_integer(value, name))
                        connection.execute(
                            "INSERT INTO command_responses "
                            "(file_id, row_index, time_ticks, "
                            "time_part1, time_part2, "
                            f"{read_sql_columns}) VALUES "
                            f"(?, ?, ?, ?, ?, {read_placeholders})",
                            (
                                file_id,
                                index,
                                seconds * 20_000_000 + clock,
                                *time_parts,
                                *values,
                            ),
                        )
    return file_id


def _ticks_to_time(ticks: np.ndarray) -> Time:
    seconds, remainder = np.divmod(ticks, _TICKS_PER_SECOND)
    return EPOCH + TimeDelta(seconds, remainder / _TICKS_PER_SECOND, format="sec")


def _photon_ticks(row, names: dict[str, str], event: bool) -> int:
    seconds = _integer(row[names["pkttimes"]], "pkttimes")
    clock = _integer(row[names["pktclock"]], "pktclock")
    ticks = seconds * _TICKS_PER_SECOND + clock
    if event:
        ticks += _integer(row[names["clocks"]], "clocks") * 256
    if not -(2**63) <= ticks < 2**63:
        raise ValueError("Photon time is outside the SQLite integer range.")
    return ticks


def _packet_key(row, names: dict[str, str]) -> tuple[int, int, int]:
    return (
        _integer(row[names["seqcount"]], "seqcount"),
        _integer(row[names["pkttimes"]], "pkttimes"),
        _integer(row[names["pktclock"]], "pktclock"),
    )


def load_photons(connection: sqlite3.Connection, filename: str | Path) -> int:
    """Import an L0 or L1 photon FITS file and return its source ID.

    Event and packet fields and FITS headers are stored separately. Identical
    files are imported once. Each event must uniquely match a packet by
    sequence count, packet seconds and packet clock within the file. The import
    is atomic and must be committed by the caller; unsupported or unmatched
    data is rejected rather than discarded.
    """
    if not connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise ValueError("Database connection must enable foreign keys.")
    path = Path(filename)
    with path.open("rb") as stream:
        digest = sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
        stream.seek(0)
        with fits.open(stream, memmap=False) as hdus:
            header = hdus[0].header
            level = str(header.get("LEVEL", "")).lower()
            if str(header.get("BTYPE", "")).lower() != "photon" or level not in {
                "l0",
                "l1",
            }:
                raise ValueError(f"{path} is not an L0 or L1 photon FITS file.")
            names = [hdu.name for hdu in hdus[1:]]
            if names.count("SCI") != 1 or names.count("PKT") != 1:
                raise ValueError(
                    "A photon FITS file must contain one SCI and one PKT extension."
                )
            for hdu in hdus[1:]:
                if (
                    hdu.name not in {"SCI", "PKT"}
                    and hdu.data is not None
                    and hdu.data.size
                ):
                    raise ValueError(f"Unsupported data extension: {hdu.name}")

            definitions = []
            for extension, table, time_column in (
                ("PKT", "photon_packets", "time_ticks"),
                ("SCI", "photon_events", "timestamp"),
            ):
                hdu = hdus[extension]
                if not isinstance(hdu, fits.BinTableHDU):
                    raise ValueError(f"{extension} must be a FITS binary table.")
                allowed = {
                    row[1] for row in connection.execute(f"PRAGMA table_info({table})")
                } - {"file_id", "row_index", "packet_row_index", time_column}
                columns = _table_columns(hdu, allowed, {})
                source_names = {target: name for name, target in columns}
                required = {"seqcount", "pkttimes", "pktclock"}
                if extension == "SCI":
                    required.add("clocks")
                if not required <= source_names.keys():
                    raise ValueError(
                        f"{extension} is missing required packet identity or timing columns."
                    )
                definitions.append((hdu, table, time_column, columns, source_names))

            packet_hdu, _, _, _, packet_names = definitions[0]
            packet_rows: dict[tuple[int, int, int], int | None] = {}
            for index, row in enumerate(packet_hdu.data):
                key = _packet_key(row, packet_names)
                packet_rows[key] = None if key in packet_rows else index

            with _import_transaction(connection):
                file_id, is_new = _store_source(
                    connection, path, digest.hexdigest(), level, hdus, "photon"
                )
                if is_new:
                    for hdu, table, time_column, columns, source_names in definitions:
                        event = table == "photon_events"
                        targets = [target for _, target in columns]
                        sql_columns = ", ".join(f'"{name}"' for name in targets)
                        placeholders = ", ".join("?" for _ in targets)
                        packet_column = "packet_row_index, " if event else ""
                        packet_placeholder = "?, " if event else ""
                        sql = (
                            f"INSERT INTO {table} (file_id, row_index, {packet_column}"
                            f"{time_column}, {sql_columns}) VALUES "
                            f"(?, ?, {packet_placeholder}?, {placeholders})"
                        )

                        for offset in range(0, len(hdu.data), 32768):
                            batch = hdu.data[offset : offset + 32768]
                            ticks = [
                                _photon_ticks(row, source_names, event) for row in batch
                            ]
                            if event:
                                times = _ticks_to_time(
                                    np.asarray(ticks, dtype=np.int64)
                                )
                                times.precision = 9
                                indexed_times = [value + "Z" for value in times.isot]
                            else:
                                indexed_times = ticks

                            def values():
                                for index, row in enumerate(batch):
                                    fields: list[int | float] = []
                                    for name, target in columns:
                                        value = row[name]
                                        if target in {"rate", "corr_rate"}:
                                            if not isinstance(
                                                value, Real
                                            ) or math.isnan(value):
                                                raise ValueError(
                                                    f"Invalid photon value for {name}."
                                                )
                                            fields.append(float(value))
                                        else:
                                            fields.append(_integer(value, name))
                                    if event:
                                        key = _packet_key(row, source_names)
                                        if key not in packet_rows:
                                            raise ValueError(
                                                f"Photon event {offset + index} has no "
                                                f"matching packet for {key}."
                                            )
                                        packet_index = packet_rows[key]
                                        if packet_index is None:
                                            raise ValueError(
                                                f"Photon event {offset + index} has "
                                                f"ambiguous packets for {key}."
                                            )
                                        yield (
                                            file_id,
                                            offset + index,
                                            packet_index,
                                            indexed_times[index],
                                            *fields,
                                        )
                                    else:
                                        yield (
                                            file_id,
                                            offset + index,
                                            indexed_times[index],
                                            *fields,
                                        )

                            connection.executemany(sql, values())
    return file_id


def _fits_series(
    rows: list[dict[str, object]],
    header_text: str,
    times: Time,
    aliases: dict[str, str],
) -> TimeSeries:
    header = fits.Header.fromstring(header_text)
    columns = []
    field_count = header.get("TFIELDS")
    if not isinstance(field_count, int) or field_count < 0:
        raise ValueError("Invalid FITS column count.")
    for index in range(1, field_count + 1):
        name = header.get(f"TTYPE{index}")
        fits_format = header.get(f"TFORM{index}")
        if not isinstance(name, str) or not isinstance(fits_format, str):
            raise ValueError(f"Invalid FITS column definition {index}.")
        fits_format = fits_format.strip().upper()
        match = re.fullmatch(r"(?:1)?([BIJKED])", fits_format)
        if match is None:
            raise ValueError(f"Unsupported FITS format: {fits_format}")
        code = match.group(1)
        zero = header.get(f"TZERO{index}")
        scale = header.get(f"TSCAL{index}")
        unsigned_zero = {"I": 2**15, "J": 2**31, "K": 2**63}.get(code)
        if scale not in (None, 1) or zero not in (None, 0, unsigned_zero):
            raise ValueError(f"Unsupported FITS scaling for column {name}.")
        dtype = {
            "B": np.uint8,
            "I": np.uint16 if zero == 2**15 else np.int16,
            "J": np.uint32 if zero == 2**31 else np.int32,
            "K": np.uint64 if zero == 2**63 else np.int64,
            "E": np.float32,
            "D": np.float64,
        }[code]
        field = aliases.get(name.lower(), name.lower())
        if code in "ED":
            values = [row[field] for row in rows]
            if any(
                not isinstance(value, Real) or math.isnan(value) for value in values
            ):
                raise ValueError(f"Invalid database value for FITS column {name}.")
        else:
            values = [_integer(row[field], field) for row in rows]
            limits = np.iinfo(dtype)
            if any(value < limits.min or value > limits.max for value in values):
                raise ValueError(f"Invalid database value for FITS column {name}.")
        columns.append(
            fits.Column(
                name=name,
                format=fits_format,
                unit=header.get(f"TUNIT{index}"),
                bzero=zero,
                bscale=scale,
                array=np.asarray(values, dtype=dtype),
            )
        )
    table = Table.read(
        fits.BinTableHDU.from_columns(columns, header=header), format="fits"
    )
    table["time"] = times
    return TimeSeries(table)


def _time_range(
    time_start: Time | str, time_end: Time | str
) -> tuple[Time, Time, Time, Time]:
    start = Time(time_start, scale="utc")
    end = Time(time_end, scale="utc")
    if not start.isscalar or not end.isscalar:
        raise ValueError("Time bounds must be scalar.")
    if any(
        np.ma.is_masked(value) for value in (start.jd1, start.jd2, end.jd1, end.jd2)
    ):
        raise ValueError("Time bounds must not be masked.")
    if not all(
        np.isfinite(value) for value in (start.jd1, start.jd2, end.jd1, end.jd2)
    ):
        raise ValueError("Time bounds must be finite.")
    if start > end:
        raise ValueError("time_start must not be later than time_end.")

    lower = start - TimeDelta(1, format="sec")
    upper = end + TimeDelta(1, format="sec")
    lower.precision = upper.precision = 9
    return start, end, lower, upper


def _timestamp_bound(time: Time, *, ceiling: bool) -> str:
    rounded = time.utc.copy()
    rounded.precision = 9
    result = Time(rounded.isot, format="isot", scale="utc")
    if ceiling and result < time:
        result += TimeDelta(1e-9, format="sec")
    elif not ceiling and result > time:
        result -= TimeDelta(1e-9, format="sec")
    result.precision = 9
    return result.isot + "Z"


def _read_binned_housekeeping(
    connection: sqlite3.Connection,
    start: Time,
    end: Time,
    num_points: int,
    aggregate: str,
) -> TimeSeries:
    columns = [
        row[1]
        for row in connection.execute("PRAGMA table_info(housekeeping)")
        if row[2].upper() in {"INTEGER", "REAL"}
        and row[1] not in {"file_id", "row_index"}
    ]
    expressions = ", ".join(
        f'{aggregate}("' + column.replace('"', '""') + '")' for column in columns
    )
    span = end - start
    times: list[Time] = []
    values: list[tuple[int | float | None, ...]] = []
    connection.execute("SAVEPOINT read_housekeeping_bins")
    try:
        for index in range(num_points if start != end else 1):
            bin_start = start + span * (index / num_points)
            bin_end = start + span * ((index + 1) / num_points)
            last_bin = index == num_points - 1 or start == end
            comparison = "<=" if last_bin else "<"
            row = connection.execute(
                f"SELECT COUNT(*), {expressions} FROM housekeeping "
                f"WHERE timestamp >= ? AND timestamp {comparison} ?",
                (
                    _timestamp_bound(bin_start, ceiling=True),
                    _timestamp_bound(bin_end, ceiling=not last_bin),
                ),
            ).fetchone()
            if row[0]:
                times.append(bin_start + (bin_end - bin_start) / 2)
                values.append(row[1:])
    finally:
        connection.execute("RELEASE SAVEPOINT read_housekeeping_bins")

    if not values:
        return TimeSeries(time=Time([], format="isot", scale="utc"))
    result = TimeSeries(time=Time(times))
    for index, column in enumerate(columns):
        column_values = [row[index] for row in values]
        result[column] = np.ma.array(
            [0 if value is None else value for value in column_values],
            mask=[value is None for value in column_values],
        )
    return result


def read_housekeeping(
    connection: sqlite3.Connection,
    time_start: Time | str,
    time_end: Time | str,
    *,
    num_points: int | None = None,
    aggregate: str = "avg",
) -> TimeSeries:
    """Return stored housekeeping samples in the inclusive time range.

    Bounds may be scalar Astropy ``Time`` objects or time strings. Results
    are sorted by time, and overlapping sources retain all their rows. An empty
    range returns an empty ``TimeSeries``. Stored times are returned unchanged;
    unlike the FITS reader, this function does not interpolate bad times.
    Set ``num_points`` to aggregate numeric database fields into at most that
    many equal-width time bins using ``aggregate`` (avg, min, max, sum, count,
    or total). Empty bins are omitted.
    """
    if not isinstance(aggregate, str) or aggregate.lower() not in {
        "avg",
        "min",
        "max",
        "sum",
        "count",
        "total",
    }:
        raise ValueError("Unsupported SQLite aggregation function.")
    aggregate = aggregate.lower()
    if num_points is None and aggregate != "avg":
        raise ValueError("num_points is required when specifying an aggregation.")
    if num_points is not None and (
        not isinstance(num_points, Integral)
        or isinstance(num_points, bool)
        or num_points < 1
    ):
        raise ValueError("num_points must be a positive integer.")
    start, end, lower, upper = _time_range(time_start, time_end)
    if num_points is not None:
        return _read_binned_housekeeping(connection, start, end, num_points, aggregate)
    cursor = connection.execute(
        "SELECT * FROM housekeeping WHERE timestamp >= ? AND timestamp <= ? "
        "ORDER BY timestamp, file_id, row_index",
        (lower.isot + "Z", upper.isot + "Z"),
    )
    names = [column[0] for column in cursor.description]
    rows = [dict(zip(names, row)) for row in cursor]
    if not rows:
        return TimeSeries(time=Time([], format="isot", scale="utc"))

    times = Time([row["timestamp"][:-1] for row in rows], format="isot", scale="utc")
    mask = (times >= start) & (times <= end)
    selected = [(row, time) for row, time, keep in zip(rows, times, mask) if keep]
    if not selected:
        return TimeSeries(time=Time([], format="isot", scale="utc"))

    grouped: dict[int, list[tuple[dict[str, object], Time]]] = {}
    for row, time in selected:
        grouped.setdefault(_integer(row["file_id"], "file_id"), []).append((row, time))

    series = []
    metadata = []
    for file_id, group in grouped.items():
        sources = connection.execute(
            "SELECT files.filename, source_extensions.header FROM files "
            "LEFT JOIN source_extensions ON files.id = source_extensions.file_id "
            "AND source_extensions.name = 'HK' WHERE files.id = ?",
            (file_id,),
        ).fetchall()
        if len(sources) != 1 or sources[0][1] is None:
            raise ValueError(f"Missing or ambiguous HK header for file {file_id}.")
        filename, header_text = sources[0]
        sample = _fits_series(
            [row for row, _ in group],
            header_text,
            Time([time for _, time in group]),
            {"timestamp": "raw_timestamp"},
        )
        metadata.append({"filename": filename, "metadata": sample.meta.copy()})
        if len(grouped) > 1:
            sample.meta.clear()
        series.append(sample)

    if len(series) == 1:
        return series[0]
    result = vstack(series, join_type="outer", metadata_conflicts="error")
    result.sort("time")
    result.meta["sources"] = metadata
    return result


def _photon_rows(
    connection: sqlite3.Connection,
    table: str,
    bounds: tuple[object, object],
) -> list[dict[str, object]]:
    time_column = "timestamp" if table == "photon_events" else "time_ticks"
    cursor = connection.execute(
        f"SELECT * FROM {table} WHERE {time_column} >= ? AND {time_column} <= ? "
        f"ORDER BY {time_column}, file_id, row_index",
        bounds,
    )
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row)) for row in cursor]


def read_photons(
    connection: sqlite3.Connection, time_start: Time | str, time_end: Time | str
) -> PhotonList:
    """Return a PhotonList with events and packet headers in the inclusive range.

    The two time series are filtered independently, matching PhotonList
    slicing. Overlapping sources retain all rows; an empty range returns a
    PhotonList with empty time series. Stored times are not trimmed.
    """
    start, end, lower, upper = _time_range(time_start, time_end)
    event_rows = _photon_rows(
        connection, "photon_events", (lower.isot + "Z", upper.isot + "Z")
    )
    minimum_tick = -(2**63)
    maximum_tick = 2**63 - 1
    lower_tick = math.floor((lower - EPOCH).to_value("sec") * _TICKS_PER_SECOND)
    upper_tick = math.ceil((upper - EPOCH).to_value("sec") * _TICKS_PER_SECOND)
    lower_tick -= _TICKS_PER_SECOND
    upper_tick += _TICKS_PER_SECOND
    if lower_tick > maximum_tick or upper_tick < minimum_tick:
        packet_rows = []
    else:
        packet_rows = _photon_rows(
            connection,
            "photon_packets",
            (max(lower_tick, minimum_tick), min(upper_tick, maximum_tick)),
        )

    selected: dict[str, dict[int, list[tuple[dict[str, object], Time]]]] = {
        "SCI": {},
        "PKT": {},
    }
    for extension, rows in (("SCI", event_rows), ("PKT", packet_rows)):
        if not rows:
            continue
        if extension == "SCI":
            timestamps = []
            for row in rows:
                timestamp = row["timestamp"]
                if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
                    raise ValueError("Invalid stored photon timestamp.")
                timestamps.append(timestamp[:-1])
            times = Time(timestamps, format="isot", scale="utc")
        else:
            ticks = np.asarray(
                [_integer(row["time_ticks"], "time_ticks") for row in rows],
                dtype=np.int64,
            )
            times = _ticks_to_time(ticks)
        mask = (times >= start) & (times <= end)
        for row, time, keep in zip(rows, times, mask):
            if keep:
                file_id = _integer(row["file_id"], "file_id")
                selected[extension].setdefault(file_id, []).append((row, time))

    file_ids = sorted(set(selected["SCI"]) | set(selected["PKT"]))
    if not file_ids:
        empty_events = TimeSeries(time=Time([], format="isot", scale="utc"))
        empty_packets = TimeSeries(time=Time([], format="isot", scale="utc"))
        return PhotonList(empty_packets, empty_events)

    series: dict[str, list[TimeSeries]] = {"SCI": [], "PKT": []}
    metadata = []
    for file_id in file_ids:
        source = connection.execute(
            "SELECT filename, primary_header FROM files WHERE id = ?", (file_id,)
        ).fetchone()
        if source is None or source[1] is None:
            raise ValueError(f"Missing primary header for photon file {file_id}.")
        filename, primary_header = source
        metadata.append(
            {
                "filename": filename,
                "metadata": dict(fits.Header.fromstring(primary_header)),
            }
        )
        for extension in ("SCI", "PKT"):
            headers = connection.execute(
                "SELECT header FROM source_extensions WHERE file_id = ? AND name = ?",
                (file_id, extension),
            ).fetchall()
            if len(headers) != 1:
                raise ValueError(
                    f"Missing or ambiguous {extension} header for file {file_id}."
                )
            group = selected[extension].get(file_id, [])
            times = (
                Time([time for _, time in group])
                if group
                else Time([], format="isot", scale="utc")
            )
            sample = _fits_series([row for row, _ in group], headers[0][0], times, {})
            if len(file_ids) > 1:
                sample.meta.clear()
            series[extension].append(sample)

    def combine(parts: list[TimeSeries]) -> TimeSeries:
        if len(parts) == 1:
            return parts[0]
        result = vstack(parts, join_type="outer", metadata_conflicts="error")
        result.sort("time")
        return result

    meta = metadata[0]["metadata"] if len(metadata) == 1 else {"sources": metadata}
    return PhotonList(combine(series["PKT"]), combine(series["SCI"]), meta=meta)
