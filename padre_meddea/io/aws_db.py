"""Provides functions to upload data to the SWxSOC time series database for dashboard display"""

import numpy as np
from astropy.timeseries import TimeSeries

from swxsoc.util.util import record_timeseries, create_annotation

from padre_meddea import log
import padre_meddea.util.util as util


def record_spectra(pkt_ts, spectra, ids):
    """Send spectrum time series data to AWS."""
    asic_nums, channel_nums = util.parse_pixelids(ids)
    # TODO: need to check that the pixelids have not changed during this time period

    NUM_LC_PER_SPEC = 4
    ADC_RANGES = np.linspace(0, 512, NUM_LC_PER_SPEC + 1, dtype=np.uint16)
    ts = TimeSeries(time=pkt_ts.time)
    for i, (this_asic, this_chan) in enumerate(zip(asic_nums[0], channel_nums[0])):
        this_col = f"Det{this_asic}{util.pixel_to_str(util.channel_to_pixel(this_chan))[:-1]}"  # remove L or S
        for j in range(NUM_LC_PER_SPEC):
            this_lc = np.sum(
                spectra.data[:, i, ADC_RANGES[j] : ADC_RANGES[j + 1]], axis=1
            )
            ts[f"{this_col.lower()}_chan{j}"] = this_lc
    record_timeseries(ts, "spectra", "meddea")
    record_timeseries(pkt_ts, "spectra_pkt", "meddea")
    create_annotation(
        pkt_ts.time[0].to_datetime(),
        f"{pkt_ts.meta['ORIGFILE']}",
        ["meta"],
        dashboard_name="MEDDEA Housekeeping",
        panel_name="Command Count",
    )


def record_photons(pkt_list, event_list):
    """Send photon time series data to AWS."""
    record_timeseries(pkt_list, "photon_pkt", "meddea")


def record_housekeeping(hk_ts):
    """Send the housekeeping time series to AWS."""
    record_timeseries(hk_ts, "housekeeping", "meddea")


def record_cmd(cmd_ts):
    """Send command time series to AWS."""
    record_timeseries(cmd_ts, "cmd_resp", "meddea")
