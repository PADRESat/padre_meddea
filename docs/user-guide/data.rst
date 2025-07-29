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

.. code-block:: python

    >>> from padre_meddea.io import read_file
    >>> ph_list = read_file("padre_meddea_l0test_photons_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

For spectrum data it returns a `~padre_meddea.spectrum.spectrum.SpectrumList`.

.. code-block:: python

    >>> from padre_meddea.io import read_file
    >>> ph_list = read_file("padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

For housekeeping data it returns two `~astropy.timeseries.TimeSeries` objects.

.. code-block:: python

    >>> from padre_meddea.io import read_file
    >>> hk_ts, cmd_ts = read_file("padre_meddea_l0test_housekeeping_20250504T070411_v0.1.0.fits")  # doctest: +SKIP

If there is no readback data then cmd_ts will be None.

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