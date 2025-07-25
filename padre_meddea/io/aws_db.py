"""Provides functions to upload data to the SWxSOC time series database for dashboard display"""

import numpy as np
from astropy.timeseries import TimeSeries

from swxsoc.util.util import record_timeseries, create_annotation

import padre_meddea.util.pixels as pixels
from padre_meddea.housekeeping.calibration import get_calibration_func


def record_spectra(pkt_ts, spectra, ids):
    """Send spectrum time series data to AWS."""
    asic_nums, channel_nums = pixels.parse_pixelids(ids)
    # TODO: need to check that the pixelids have not changed during this time period

    NUM_LC_PER_SPEC = 4
    ADC_RANGES = np.linspace(0, 512, NUM_LC_PER_SPEC + 1, dtype=np.uint16)
    ts = TimeSeries(time=pkt_ts.time)
    median_asic_nums = np.median(asic_nums, axis=0)
    median_channel_nums = np.median(channel_nums, axis=0)
    for i, (this_asic, this_chan) in enumerate(
        zip(median_asic_nums, median_channel_nums)
    ):
        this_col = f"Det{this_asic}{pixels.pixel_to_str(pixels.channel_to_pixel(this_chan))[:-1]}"  # remove L or S
        for j in range(NUM_LC_PER_SPEC):
            this_lc = np.sum(
                spectra.data[:, i, ADC_RANGES[j] : ADC_RANGES[j + 1]], axis=1
            )
            ts[f"{this_col.lower()}_chan{j}"] = this_lc
    record_timeseries(ts, "spectra", "meddea")
    record_timeseries(pkt_ts, "spectra_pkt", "meddea")
    create_annotation(pkt_ts.time[0], f"{pkt_ts.meta['ORIGFILE']}", ["meta"])


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
