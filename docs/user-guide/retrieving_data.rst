.. _retrieving_data:

**********************************************
Accessing PADRE MeDDEA Data with PADREClient
**********************************************

Introduction
============

The PADRE MeDDEA instrument data can be accessed using the :class:`~padre_meddea.net.PADREClient`, a custom client that integrates with SunPy's Fido interface.
This client allows you to search and download MeDDEA data from the Solar Data Analysis Center (SDAC) servers.

PADREClient and DataType
========================

The :class:`~padre_meddea.net.PADREClient` is specifically designed to work with data from the PADRE MeDDEA instrument. 
It handles various data types and processing levels, making it easier to access the data you need for your research.

Additionally, a :class:`~padre_meddea.net.DataType` attribute class is provided for specifying the type of data you want to search for:

.. code-block:: python

    from padre_meddea.net import PADREClient, DataType
    from sunpy.net import Fido
    from sunpy.net import attrs as a

Attributes for Searching Data
=============================

The `PADREClient` supports the following search attributes:

- **`a.Time`**: The time range for the data (e.g., `a.Time("2025-05-01", "2025-05-05")`)
- **`a.Instrument`**: The instrument name (use `a.Instrument.meddea` for MeDDEA data)
- **`a.Level`**: The data processing level (`a.Level.raw`, `a.Level.l0`, `a.Level.l1`)
- **`DataType`**: The type of data (`DataType.spectrum`, `DataType.photon`, `DataType.housekeeping`)

Examples for Searching Data
===========================

Example 1: Searching for Spectrum Data
--------------------------------------

To search for spectrum data across all available levels:

.. code-block:: python

    results = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & DataType.spectrum
    )
    results

Example 2: Searching for Level 1 Data
-------------------------------------

To search for all Level 1 data regardless of data type:

.. code-block:: python

    results = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & a.Level.l1
    )
    results

Example 3: Searching for Level 1 Photon Data
--------------------------------------------

To search for Level 1 photon data specifically:

.. code-block:: python

    results = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & 
        a.Instrument.meddea & 
        a.Level.l1 & 
        DataType.photon
    )
    results

Example 4: Searching for Raw Data
----------------------------------

To search for all raw data:

.. code-block:: python

    results = Fido.search(
        a.Time("2025-05-01", "2025-05-05") & a.Instrument.meddea & a.Level.raw
    )
    results

Example 5: Searching for Raw Housekeeping Data
----------------------------------------------

To search for raw housekeeping data specifically:

.. code-block:: python

    results = Fido.search(
        a.Time("2025-05-01", "2025-05-05") &
        a.Instrument.meddea &
        a.Level.raw &
        DataType.housekeeping
    )
    results

Downloading Data
================

After performing a search, you can download the data using the standard Fido interface:

.. code-block:: python

    import tempfile

    # Create a temporary directory to store downloaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        downloaded_files = Fido.fetch(results, path=temp_dir)
    downloaded_files

You can also specify a permanent location for the files:

.. code-block:: python

    downloaded_files = Fido.fetch(results, path="./my_data_dir/")
