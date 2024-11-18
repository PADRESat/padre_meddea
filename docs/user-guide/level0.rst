.. _level0:

************
Level 0 Data
************

Overview
========
Level 0 data are provided in the `FITS file <https://fits.gsfc.nasa.gov/>`__ format.
For more information on how to read or write FITS file see `astropy.fits <https://docs.astropy.org/en/stable/io/fits/index.html>`__.
This section describes the organization the level 0 FITS files.
Level 0 fits files generally include the unconverted data from the raw binary files of ccsds packets.
The purpose of these files is to provide the raw data from the raw binary files in a more convinient form for analysis.
It also provides metadata information which summary the data in the file.

Level 0 event files
-------------------

This file contains the data from all events that triggered the detectors.
They consist of 3 HDUs including the primary HDU.
The primary HDU contains no data and is only used for metadata.
The two other HDUs are named `SCI` and `PKT`.
`SCI` includes the event data while `PKT` includes the packet data.
Each data packet may include one of more event therefore there is a one to many relationship between them.
In order to understand the relationship between the events and packets, each event provides the associated packet sequence number.
This sequence number can be used to lookup the packet data for that event.

Primary HDU
***********
No data is provided.
Stay tuned for a list of metadata 

PKT HDU
*******
The following columns are provided for each data packet.
The bits column provide the number of significant bits and not the bit length of the column itself.
The columns in the FITS file are provided in the smallest possible data type.

======== ============================================= ====
name                                                   bits
======== ============================================= ====
seqcount packet sequence number, should be consecutive   12
pkttimes the packet time in seconds since EPOCH          32
pktclock the packet subsecond time in clocks             32
livetime live time                                       16
inttime  integration time in real time                   16
flags    flags                                           16
======== ============================================= ====

SCI HDU
*******
The following columns are provided for each event or photon detected.
The bits column provide the number of significant bits and not the bit length of the column itself.
The columns in the FITS file are provided in the smallest possible data type.

======== ============================================================================================ ====
name     description                                                                                  bits
======== ============================================================================================ ====
seqcount packet sequence number                                                                       12
clocks   the clock number                                                                             16
asic     the asic number or detector id                                                                3
channel  the asic channel which is related to the pixel                                                5
atod     the uncalibrated energy of the event in ADC counts                                           12
baseline the baseline measurement if exists, otherwise all zeros                                      12
pkttimes the packet time in seconds since EPOCH, also exists in PKT,                                  32
pktclock the packet time in clocks since EPOCH, also exists in PKT                                    32
======== ============================================================================================ ====

Level 0 spectrum files
----------------------
Summary spectra are created for 24 pixels at a regular cadence (normally every 10 s)
Each spectrum has a total of 512 energy bins.

Level 0 housekeeping files
--------------------------
These files contain housekeeping data as described in the housekeeping packet.
It also includes any register read responses that may exist during that time period.
