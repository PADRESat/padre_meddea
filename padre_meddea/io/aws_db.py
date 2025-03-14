"""Provides functions to upload data to the SWxSOC time series database for dashboard display"""

import numpy as np
from astropy.timeseries import TimeSeries

from swxsoc.util.util import record_timeseries, create_annotation

from padre_meddea import log
import padre_meddea.util.util as util


def record_spectra(ts, spectra, ids):
    """Send spectrum time series data to AWS."""
    asic_nums, channel_nums = util.parse_pixelids(ids)
    # TODO: need to check that the pixelids have not changed during this time period

    # create timeseries for each spectrum
    NUM_LC_PER_SPEC = 4
    ADC_RANGES = np.linspace(0, 512, NUM_LC_PER_SPEC + 1, dtype=np.uint16)

    for i, (this_asic, this_chan) in enumerate(zip(asic_nums[0], channel_nums[0])):
        this_col = (
            f"Det{this_asic}{util.pixel_to_str(util.channel_to_pixel(this_chan))}"
        )
        log.info(
            f"Sending spectrum {i:02} {this_col} lightcurve to timestream to table spec{i:02}"
        )
        ts = TimeSeries(time=ts.time)
        for j in range(NUM_LC_PER_SPEC):
            this_lc = spectra.data[:, i, ADC_RANGES[j] : ADC_RANGES[j + 1]]
            ts[f"channel{j}"] = this_lc
    record_timeseries(ts, "spectra", "meddea")
    create_annotation(ts.time[0], f"{ts.meta['ORIGFILE']}", ["meta"])


def record_photons(pkt_list, event_list):
    """Send photon time series data to AWS."""
    record_timeseries(pkt_list, "photon_pkt", "meddea")
    create_annotation(pkt_list.time[0], f"{pkt_list.meta['ORIGFILE']}", ["meta"])


def record_housekeeping(hk_ts):
    """Send the housekeeping time series to AWS."""
    record_timeseries(hk_ts, "housekeeping", "meddea")
    create_annotation(hk_ts.time[0], f"{hk_ts.meta['ORIGFILE']}", ["meta"])


def record_cmd(cmd_ts):
    """Send command time series to AWS."""
    record_timeseries(cmd_ts, "cmd_resp", "meddea")
    create_annotation(cmd_ts.time[0], f"{cmd_ts.meta['ORIGFILE']}", ["meta"])
