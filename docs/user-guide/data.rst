.. _data:

****
Data
****

Data Description
----------------
MeDDEA has two primary science data products which originate from the same measurements.

#. an x-ray spectrum, provided regularly and
#. a x-ray photon list provided on-demand.

Generally, the photon list data product will only be available during large flares and calibration periods.
It is generated automatically on board the spacecraft but only downloaded when requested.

MeDDEA also generates housekeeping data products.

All MeDDEA data will be made publicly available `here <https://umbra.nascom.nasa.gov/padre/padre-meddea/>`_.

Raw data consists of photon data, spectrum data, and housekeeping data in its raw binary form.
Level 0 data consists of the same data as the raw data but in fits files.
Each raw data file generates a single level 0 data file.
Level 1 data concatenate level 0 data files into daily files or flare files.

The data levels for the spectrum product are described below.
These products are processed on the ground using this Python package unless otherwise specified.

Reading data
------------
There is one primary way to read in data files `~padre_meddea.io.read_file()`.
This function can read both binary (.DAT) and fits files (.fits).

For photons data it returns a `~padre_meddea.spectrum.spectrum.PhotonList`.

.. doctest:: python

    >>> from padre_meddea.io import read_file
    >>> ph_list = read_file("padre_meddea_l0test_photons_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

For spectrum data it returns a `~padre_meddea.spectrum.spectrum.SpectrumList`.

.. doctest:: python

    >>> from padre_meddea.io import read_file
    >>> ph_list = read_file("padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

For housekeeping data it returns two `~astropy.timeseries.TimeSeries` objects.

.. doctest:: python

    >>> from padre_meddea.io import read_file
    >>> hk_ts, cmd_ts = read_file("padre_meddea_l0test_housekeeping_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

If there is no readback data then cmd_ts will be None.

Local database
--------------

Create or open a local SQLite database and import a housekeeping FITS file:

.. code-block:: python

    from padre_meddea.db import connect, load_housekeeping

    connection = connect("meddea.sqlite")
    try:
        file_id = load_housekeeping(connection, "housekeeping.fits")
        connection.commit()
    finally:
        connection.close()

`~padre_meddea.db.load_housekeeping` imports L0 or L1 HK and READ tables
atomically, without committing the transaction. Re-importing identical file
contents returns the existing source ID. Unsupported columns, FITS null
markers, and additional nonempty data extensions are rejected instead of
silently discarded.

Retrieve housekeeping samples as an Astropy ``TimeSeries``:

.. code-block:: python

    from padre_meddea.db import connect, read_housekeeping

    connection = connect("meddea.sqlite")
    try:
        hk = read_housekeeping(connection, "2025-05-04T05:51:38", "2025-05-04T06:00:00")
    finally:
        connection.close()

The time bounds are inclusive and can also be scalar Astropy ``Time`` objects.
Rows from overlapping source files are retained; columns absent from one file
are masked. For multiple sources, ``hk.meta["sources"]`` contains their
filenames and per-file metadata. A range without samples returns an empty
``TimeSeries``. Stored
housekeeping timestamps have nanosecond precision and are returned as stored:
unlike the FITS reader, the query does not interpolate bad times. Command
responses are not included.

To reduce the number of returned points, use SQLite aggregation over equal-width
time bins:

.. code-block:: python

    connection = connect("meddea.sqlite")
    try:
        hk_binned = read_housekeeping(
            connection, "2025-05-04T05:51:38", "2025-05-04T06:00:00",
            num_points=100, aggregate="avg",
        )
    finally:
        connection.close()

``num_points`` is a positive maximum number of output bins. Empty bins are
omitted; each returned time is the bin midpoint. Bins include their start and
exclude their end, except that the final bin includes the query end. Supported
aggregates are ``avg``, ``min``, ``max``, ``sum``, ``count``, and ``total``.
All numeric housekeeping database fields are aggregated, including counters
and flags, so select a function appropriate for the fields being examined.
Results use database column names (for example, ``raw_timestamp``), not the
original FITS column names, units, or metadata. SQLite ignores NULL values:
all-NULL fields are masked for ``avg``, ``min``, ``max``, and ``sum``; ``count``
and ``total`` return zero. The unbinned result is unchanged when ``num_points``
is omitted.

Import photon FITS data and retrieve a ``PhotonList``:

.. code-block:: python

    from padre_meddea.db import connect, load_photons, read_photons

    connection = connect("meddea.sqlite")
    try:
        load_photons(connection, "photon.fits")
        connection.commit()
        photons = read_photons(connection, "2026-07-04T20:40:00", "2026-07-04T20:41:00")
    finally:
        connection.close()

Photon imports are atomic and deduplicated by file checksum. Unsupported FITS
columns, null markers, extra nonempty data extensions and NaN packet rates are
rejected. Each event references exactly one packet from the same source, matched
using sequence count, packet seconds and packet clock. Files with missing or
ambiguous matches are rejected. The inclusive query filters event and packet
times independently, so a packet referenced by an event can lie outside the
returned range. The query retains overlapping sources and sorts each time
series. A range without data returns a ``PhotonList`` with empty time series.
Photon times are calculated
from instrument clocks at 50 ns resolution, without the FITS reader's
floating-point rounding or its trimming of old timestamps. Multiple sources
are described in ``photons.meta["sources"]``.

The database has separate photon event and packet tables, mirroring the FITS
``SCI`` and ``PKT`` extensions, as well as housekeeping and command-response
tables. Rows retain their source and original row position.
Every event, packet, housekeeping sample and command response has a non-null
``file_id`` foreign key to the single originating file in ``files``.
Use ``connect`` to enable SQLite's foreign-key enforcement. The optional
``timestamp`` columns in photon events and housekeeping contain UTC ISO 8601
text with nine fractional digits, such as ``2026-07-04T20:40:13.123456750Z``.
SQLite has no native time storage type; the fixed-width UTC format supports
chronological text comparisons without losing 50 ns resolution. SQLite checks
the timestamp's format, not whether it is a valid UTC date; callers must
validate the date before inserting it. The original housekeeping FITS
``timestamp`` field is retained as ``raw_timestamp``.
The photon packet and command-response ``time_ticks`` columns represent elapsed
SI time in 50 ns ticks since 2000-01-01 00:00:00 UTC. They must be left NULL
when a time cannot be represented exactly.
FITS extension headers can be retained in ``source_extensions``; the two
components of a command-response FITS ``time`` column correspond to
``time_part1`` and ``time_part2``.
Databases with incompatible schema versions are rejected.

Calibrated data
---------------

Coming soon! The following is not yet implemented.


Both of the above products can be used to generate a calibrated spectrum product.

+----------+---------------------------------------+---------------------------------------+
| Level    | Product                               | Description                           |
+==========+=======================================+=======================================+
| 2        | Flux Spectrum in energy space,        | FITS, data flag to state if it was    |
|          | integrated over all detectors and     | generated from the photon list or not |
|          | pixels                                |                                       |
+----------+---------------------------------------+---------------------------------------+

The above data product will be used to generate the following derived data products.

+----------+---------------------------------------+---------------------------------------+
| Level    | Product                               | Description                           |
+==========+=======================================+=======================================+
| 3        | Flare X-ray Directivity as a function | FITS file, requires Solar Orbiter STIX|
|          | of energy and time for the angular    | data, ratio of STIX to PADRE flux     |
|          | separation between STIX and PADRE     |                                       |
+----------+---------------------------------------+---------------------------------------+
| 4        | Flare-accelerated Electron Anisotropy | FITS file, requires modeling analysis |
|          | as a function of energy and time.     |                                       |
+----------+---------------------------------------+---------------------------------------+

File Naming Conventions
-----------------------

The file naming conventions for the products listed above are

Raw data

* Event data PADREMDA0_250504055133.DAT
* Spectrum data PADREMDA2_250504070426.DAT
* Housekeeping data PADREMDU8_250504153121.DAT

Level 0 data

* Event data padre_meddea_l0_photon_20250504T153114_v0.1.0.fits
* Spectrum data padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits
* Housekeeping data padre_meddea_l0_housekeeping_20250504T055138_v0.1.0.fits

Level 1 data

* padre_meddea_l1_photon_20250504T000000_v0.1.0.fits
* padre_meddea_l1_spectrum_20250504T000000_v0.1.0.fits
* padre_meddea_l1_housekeeping_20250504T000000_v0.1.0.fits	2025