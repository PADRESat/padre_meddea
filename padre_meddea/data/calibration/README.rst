Calibration directory
=====================

This directory contains calibration files included with the package source
code distribution.

Naming Convention
-----------------
Filenames are structed as followed

  MEDDEA_<instrument>_calib_<start time>_<end_time>

For example,

  padre_meddea_calib_20220401_20220501


padre_meddea_calib_hk_0.csv
---------------------------
The calibration file file for housekeeping data.

Contains calibration function data to convert housekeeping values from measured values to physical values.
The columns are defined as follows:
  * hk_name: the string identifier for the housekeeping field
  * cal_func: the calibration function type (e.g. poly = np.poly1d)
  * cal_params: the calibration function parameters (e.g. for poly functions "1, 2, 3" means 1 x**2 + 2 x + 3)
  * phys_unit: the physical unit of the measurement after calibration. Must be given as an astropy.units compatible string.

