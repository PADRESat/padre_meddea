"""Provides functions to upload data to the SWxSOC time series database for dashboard display"""

import astropy.units as u
import numpy as np
from astropy.timeseries import TimeSeries
from specutils import SpectralRegion
from swxsoc.util.util import create_annotation, record_timeseries

from padre_meddea.housekeeping.calibration import get_calibration_func
from padre_meddea.spectrum.spectrum import SpectrumList


def record_spectra(spec_list: SpectrumList):
    """Send spectrum time series data to AWS."""
    if spec_list.calibrated:
        sr = SpectralRegion([[5, 10], [10, 15], [15, 25], [25, 50], [50, 100]] * u.keV)
        ts = spec_list.lightcurve(pixel_list=spec_list.pixel_list, sr=sr)
        record_timeseries(ts, "spectra_sum", "meddea")
        ts_specgram = spec_list.spectrogram()
        record_timeseries(ts_specgram, "spectragram", "meddea")


def record_photons(pkt_list, event_list):
    """Send photon time series data to AWS."""
    record_timeseries(pkt_list, "photon_pkt", "meddea")
    create_annotation(pkt_list.time[0], f"{pkt_list.meta['ORIGFILE']}", ["meta"])


def record_housekeeping(hk_ts: TimeSeries):
    """Send the housekeeping time series to AWS."""
    my_hk_ts = hk_ts.copy()
    colnames_to_remove = [
        "CCSDS_APID",
        "CCSDS_VERSION_NUMBER",
        "CCSDS_PACKET_TYPE",
        "CCSDS_SECONDARY_FLAG",
        "CCSDS_SEQUENCE_FLAG",
        "CCSDS_SEQUENCE_COUNT",
        "CCSDS_PACKET_LENGTH",
        "timestamp",
        "CHECKSUM",
    ]
    for this_col in colnames_to_remove:
        if this_col in hk_ts.colnames:
            my_hk_ts.remove_column(this_col)
    # calibrate hard to calibrate columns before sending
    colnames_to_calibrate = ["fp_temp", "hvps_temp", "dib_temp"]
    for this_col in colnames_to_calibrate:
        if this_col in hk_ts.colnames:
            f = get_calibration_func(this_col)
            my_hk_ts[f"cal_{this_col}"] = f(hk_ts[this_col]).value

    record_timeseries(my_hk_ts, "housekeeping", "meddea")
    create_annotation(my_hk_ts.time[0], f"{hk_ts.meta['ORIGFILE']}", ["meta"])


def record_cmd(cmd_ts):
    """Send command time series to AWS."""
    record_timeseries(cmd_ts, "cmd_resp", "meddea")
    create_annotation(cmd_ts.time[0], f"{cmd_ts.meta['ORIGFILE']}", ["meta"])
