.. _data:

****
Data
****

Overview
========

Data Description
----------------
MeDDEA has two primary data products which originate from the same measurements.

#. an x-ray spectrum, provided regularly and 
#. a x-ray photon list provided on-demand.

Generally, the photon list data product will only exist during large flares and calibration periods.
It is generated automatically on board the spacecraft but only downloaded when requested.
This photon list will be used to generate an x-ray spectrum that supersedes the histogram data product.

The data levels for the spectrum product are described below.
These products are processed on the ground using this Python package unless otherwise specified.

+----------+---------------------------------------+---------------------------------------+
| Level    | Product                               | Description                           |      
+==========+=======================================+=======================================+
| 1        | Count Spectrum in energy space        | FITS file, produced at least every 1 s|
|          | integrated across all pixels and      | , generated on the spacecraft         |
|          | detectors                             |                                       |
+----------+---------------------------------------+---------------------------------------+

The data levels for the photon list product are described below.

+----------+---------------------------------------+---------------------------------------+
| Level    | Product                               | Description                           |      
+==========+=======================================+=======================================+
| 0        | List of hits. Each photon has         | FITS file, file will consist of a     |
|          | relative time, photon energy in ADC   | fixed number of hits                  |
|          | counts and pixel and detector number  | , generated on the spacecraft         |
+----------+---------------------------------------+---------------------------------------+
| 1        | List of photons. Each photon has      | FITS file, produced with a fixed      |
|          | time of arrival in UTC and calibrated | number photons and variable           |                       
|          | energy for each pixel and detector    | integration times.                    |
+----------+---------------------------------------+---------------------------------------+

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

#. padre_meddea_l0_photonlist_%Y%m%d_v{version}
#. padre_meddea_l1_spec_%Y%m%d_v{version}
#. padre_meddea_l2_spec_%Y%m%d_v{version}
#. padre_meddea_l3_xraydirect_%Y%m%d_{angle}_v{version}
#. padre_meddea_l4_eanisotropy_%Y%m%d_v{version}

Files will be generated daily and include a full day of data.
The `{version}` begins at one and is incremented everytime the data file is updated.
The `{angle}` is the average angular separation between STIX and PADRE at the time of the observation.
If too large, a file may be split up into multiple files in which case the date string will add the hour and minute and the integration time.
Specified using Python datetime strftime definitions and version is a 3 digit zero-padded number which begins at 000 and increments every time the file is reprocessed.
See `Python strftime cheatsheet <https://strftime.org/>`_ for a quick reference.

Getting Data
============

To be written.

Reading Data
============

Calibrating Data
================
Data products below level 2 generally require calibration to be transformed into scientificically useable units.
This section describes how to calibrate data files from lower to higher levels.