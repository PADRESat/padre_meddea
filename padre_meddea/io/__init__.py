import numpy as np
from astropy.io import ascii
from astropy.table import Table

import padre_meddea

detector_values = Table(ascii.read(padre_meddea._data_directory / "detector_values.csv"))

#  a dictionary to convert asic channel numbers to pixel numbers
ASIC_CHANNEL_TO_PIXEL = {}
for this_asic_num in np.unique(detector_values['asic_num']):
    this_array = list(detector_values[detector_values['asic_num'] == this_asic_num]['asic_channel'])
    ASIC_CHANNEL_TO_PIXEL.update({this_asic_num: this_array})
