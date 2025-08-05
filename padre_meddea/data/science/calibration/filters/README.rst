Filter Calibration Files
========================

This directory contains files to be used to calibrate the elements in the optical path.
This includes
    * Beryllium alloy filter
    * Aluminum filters (2x), each 0.2 mm thick
    * Thermal blankets (1 layer of silver teflon)

The measurements were taken by shining x-rays through the material using an x-ray tube.
The unfiltered and filtered (or attenuated) data files are provided here.

Each filename contains the voltage and current settings used for each measurement, the type of measurement (i.e. filtered or unfiltered), and the ASIC corresponding to the window through which the measurements were made.

Filenaming convention:
YYYYMMDD_HHMMSS_{minix voltage}kV_{Mini-X current}mA_Bewin{asic #}_{position #}.mca

Please refer to this calibration document for further details:
https://docs.google.com/document/d/1MET3BjQ_a0BxfH67vQB_gVwgNtqqsfqxdSTeWtHttCQ/edit?tab=t.0
