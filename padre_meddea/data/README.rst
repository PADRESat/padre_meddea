Data directory
==============

This directory contains data files included with the package source
code distribution. Note that this is intended only for relatively small files
- large files should be externally hosted and downloaded as needed.
This directory is NOT for downloaded instrument data unless it is needed for testing purposes.
Note that calibration files are stored in the `calibration` subdirectory.

ba133.csv
---------
This file contains the x-ray emission lines from radioactive Barium-133.
Source: http://nucleardata.nuclear.lu.se/toi/nuclide.asp?iZA=560133
See also
- http://www.lnhb.fr/nuclides/Ba-133_tables.pdf
- Grimm, O., Spectral signature of near-surface damage in CdTe X-ray detectors, Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment, Volume 953, 2020, 163104, ISSN 0168-9002,
https://doi.org/10.1016/j.nima.2019.163104


SOL2002-07-23_RHESSI_flare_spectrum.csv
---------------------------------------
The x-ray spectrum of the SOL2002-07-23 X4.8 class flare as observed by RHESSI (Lin et al. 2002).
Spectrum courtesy of Dr. Albert Shih (@ayshih).

SOL2002-07-23_GOESXRS_lightcurve.csv
------------------------------------
The lightcurve of the SOL2002-07-23 X4.8 class flare as observed by GOES XRS.
These data were downloaded using sunpy.
Data drop outs were manually removed.

fits_keywords_hdu0.csv
----------------------
Default keyword values for the primary HDU for all MeDDEA FITS files.

detector_values.csv
-------------------
Stores detector constants.

hk_channel_defs.csv
-------------------
Stores the definitions for the values provided in housekeeping packets.

fits_