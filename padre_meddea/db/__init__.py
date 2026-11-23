"""Initialize a local SQLite database for MeDDEA data."""

import sqlite3
from collections.abc import Iterator
from importlib.resources import files
from pathlib import Path

from .db import load_housekeeping, load_photons, read_housekeeping, read_photons

__all__ = [
    "connect",
    "load_housekeeping",
    "load_photons",
    "read_housekeeping",
    "read_photons",
]

_APPLICATION_ID = 0x4D444442
_SCHEMA_VERSION = 4


def _schema_statements() -> Iterator[str]:
    schema = files("padre_meddea.db").joinpath("schema.sql").read_text(encoding="utf-8")
    statement = ""
    for line in schema.splitlines(keepends=True):
        statement += line
        if sqlite3.complete_statement(statement):
            yield statement
            statement = ""
    if statement.strip():
        raise ValueError("Incomplete SQL statement in database schema.")


def connect(path: str | Path) -> sqlite3.Connection:
    """Open or initialize a MeDDEA database.

    The caller owns the returned connection and must close it. Pass ``":memory:"``
    for an in-memory database. Incompatible schemas and unrelated contents
    are rejected.
    """
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        application_id = connection.execute("PRAGMA application_id").fetchone()[0]
        version = connection.execute("PRAGMA user_version").fetchone()[0]

        if application_id == 0 and version == 0:
            existing_objects = connection.execute(
                "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' LIMIT 1"
            ).fetchone()
            if existing_objects:
                raise ValueError("Cannot initialize a non-empty SQLite database.")

            for statement in _schema_statements():
                connection.execute(statement)
            connection.execute(f"PRAGMA application_id = {_APPLICATION_ID}")
            connection.execute(f"PRAGMA user_version = {_SCHEMA_VERSION}")
        elif application_id != _APPLICATION_ID or version != _SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported database: application_id={application_id}, "
                f"schema version={version}."
            )
        connection.commit()
    except Exception:
        connection.close()
        raise
    return connection
