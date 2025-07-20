"""Provides support for parsing raw files"""

from pathlib import Path

import astropy.units as u
import ccsdspy
import numpy as np
from astropy.timeseries import TimeSeries
from ccsdspy.utils import split_by_apid
from specutils import Spectrum1D

import padre_meddea
from padre_meddea import log
import padre_meddea.util.util as util
import padre_meddea.util.pixels as pixels


def parse_ph_packets(filename: Path):
    """Given a binary file, read only the photon packets and return an event list.

    Photon packets consist of 15 header words which includes a checksum.
    Each photon adds 3 words.

    Photon packet format is (words are 16 bits) described below.

    ==  =============================================================
    #   Description
    ==  =============================================================
    0   CCSDS header 1 (0x00A0)
    1   CCSDS header 2 (0b11 and sequence count)
    2   CCSDS header 3 payload size (remaining packet size - 1 octet)
    3   time_stamp_s 1
    4   time_stamp_s 2
    5   time_stamp_clocks 1
    6   time_stamp_clocks 2
    7   integration time in clock counts
    8   live time in clock counts
    9   flags, drop counter ([15] Int.Time Overflow, [14:12] decimation level, [11:0] # dropped photons)
    10  checksum
    11  start of pixel data (each is 16 bit field)
    -   pixel time step in clock count
    -   pixel_id (ASIC # bits[7:5], pixel num bits[4:0])
    -   pixel_data ADC count
    -   pixel_data baseline 12 bit baseline value (optional)
    ==  =============================================================

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    ph_list : astropy.time.TimeSeries or None
        A photon list
    """

    filename = Path(filename)
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_stream = stream_by_apid.get(padre_meddea.APID["photon"], None)
    if packet_stream is None:
        return None
    else:
        padre_meddea.log.info(f"{filename.name}: Found photon data")
    packet_definition = packet_definition_ph()
    pkt = ccsdspy.VariableLength(packet_definition)
    ph_data = pkt.load(packet_stream, include_primary_header=True)
    # packet_time = ph_data["TIME_S"] + ph_data["TIME_CLOCKS"] / 20.0e6
    if util.has_baseline(filename):
        padre_meddea.log.info(
            f"{filename.name}: Found baseline measurements in photon data."
        )
        WORDS_PER_HIT = 4
    else:
        padre_meddea.log.info(
            f"{filename.name}: No baseline measurements in photon data."
        )
        WORDS_PER_HIT = 3

    pkt_times = util.calc_time(ph_data["TIME_S"], ph_data["TIME_CLOCKS"])
    pkt_list = TimeSeries(time=pkt_times)
    pkt_list["seqcount"] = ph_data["CCSDS_SEQUENCE_COUNT"]
    pkt_list["pkttimes"] = ph_data["TIME_S"]
    pkt_list["pktclock"] = ph_data["TIME_CLOCKS"]
    pkt_list["livetime"] = ph_data["LIVE_TIME"]
    pkt_list["inttime"] = ph_data["INTEGRATION_TIME"]
    pkt_list["flags"] = ph_data["FLAGS"]
    # parse flag field into their own
    (
        pkt_list["decim_lvl"],
        pkt_list["drop_cnt"],
        pkt_list["int_time_flag"],
    ) = util.parse_ph_flags(pkt_list["flags"])
    pkt_list.meta.update({"ORIGFILE": f"{filename.name}"})

    # determine the total amount of hits in all photon packets
    hit_count = np.zeros(len(pkt_list), dtype="uint32")
    for i, this_packet in enumerate(ph_data["PIXEL_DATA"]):
        hit_count[i] = len(this_packet[0::WORDS_PER_HIT])
    total_hits = np.sum(hit_count)
    # add the number of hits per photon packet to pkt_list
    pkt_list["num_hits"] = hit_count
    pkt_list["rate"] = hit_count / (pkt_list["inttime"] * 12.8e-6)
    pkt_list["corr_rate"] = hit_count / (pkt_list["livetime"] * 12.8e-6)

    if (total_hits - np.floor(total_hits)) != 0:
        raise ValueError(
            f"Got non-integer number of hits {total_hits - np.floor(total_hits)}."
        )

    # hit_list definition
    # 0, raw id number
    # 1, asic number, 0 to 7
    # 2, channel number, 0 to 32
    # 3, energy 12 bits
    # 4, packet sequence number 12 bits
    # 5, clock number
    # 6, baseline (if exists) otherwise 0

    hit_list = np.zeros((9, total_hits), dtype="uint16")
    time_s = np.zeros(total_hits, dtype="uint32")
    time_clk = np.zeros(total_hits, dtype="uint32")

    # 0 time, 1 asic_num, 2 channel num, 3 hit channel
    i = 0
    # iterate over packets
    for this_pkt_num, this_ph_data, pkt_s, pkt_clk in zip(
        ph_data["CCSDS_SEQUENCE_COUNT"],
        ph_data["PIXEL_DATA"],
        ph_data["TIME_S"],
        ph_data["TIME_CLOCKS"],
    ):
        num_hits = len(this_ph_data[0::WORDS_PER_HIT])
        ids = this_ph_data[1::WORDS_PER_HIT]
        asic_num = (ids & 0b11100000) >> 5
        channel_num = ids & 0b00011111
        hit_list[0, i : i + num_hits] = this_ph_data[0::WORDS_PER_HIT]
        hit_list[1, i : i + num_hits] = asic_num
        hit_list[2, i : i + num_hits] = channel_num
        hit_list[4, i : i + num_hits] = this_pkt_num
        hit_list[5, i : i + num_hits] = this_ph_data[0::WORDS_PER_HIT]
        time_s[i : i + num_hits] = pkt_s
        time_clk[i : i + num_hits] = pkt_clk
        if WORDS_PER_HIT == 4:
            hit_list[6, i : i + num_hits] = this_ph_data[2::WORDS_PER_HIT]  # baseline
            hit_list[3, i : i + num_hits] = this_ph_data[3::WORDS_PER_HIT]  # hit energy
        elif WORDS_PER_HIT == 3:
            hit_list[3, i : i + num_hits] = this_ph_data[2::WORDS_PER_HIT]  # hit energy
        i += num_hits

    ph_times = util.calc_time(
        time_s,
        time_clk,
        hit_list[5, :],
    )

    event_list = TimeSeries(time=ph_times)
    event_list["seqcount"] = hit_list[4, :]
    event_list["clocks"] = hit_list[5, :]
    event_list["asic"] = hit_list[1, :].astype(np.uint8)
    event_list["channel"] = hit_list[2, :].astype(np.uint8)
    event_list["atod"] = hit_list[3, :]
    event_list["baseline"] = hit_list[6, :]  # if baseline not present then all zeros
    event_list["pkttimes"] = time_s
    event_list["pktclock"] = time_clk
    event_list["pixel"] = pixels.channel_to_pixel(event_list["channel"])
    date_beg = util.calc_time(time_s[0], time_clk[0])
    date_end = util.calc_time(time_s[-1], time_clk[-1])
    event_list.meta.update({"DATE-BEG": date_beg.fits})
    event_list.meta.update({"DATE-END": date_end.fits})
    event_list.meta.update({"ORIGFILE": f"{filename.name}"})
    center_index = int(len(time_s) / 2.0)
    date_avg = util.calc_time(time_s[center_index], time_clk[center_index])
    event_list.meta.update({"DATE-AVG": date_avg.fits})

    return pkt_list, event_list


def parse_spectrum_packets(filename: Path):
    """Given a binary file, read only the spectrum packets and return the data.

    Parameters
    ----------
    filename : Path
        A file to read

    Returns
    -------
    hk_list : astropy.time.TimeSeries or None
        A list of spectra data or None if no spectrum packets are found.
    """
    import ccsdspy
    from ccsdspy.utils import split_by_apid

    filename = Path(filename)
    with open(filename, "rb") as mixed_file:
        stream_by_apid = split_by_apid(mixed_file)
    packet_bytes = stream_by_apid.get(padre_meddea.APID["spectrum"], None)
    if packet_bytes is None:
        return None
    else:
        log.info(f"{filename.name}: Found spectrum data")
    packet_definition = packet_definition_hist2()
    pkt = ccsdspy.FixedLength(packet_definition)
    data = pkt.load(packet_bytes, include_primary_header=True)
    timestamps = util.calc_time(data["TIME_S"], data["TIME_CLOCKS"])
    num_packets = len(data["TIME_S"])
    h = data["HISTOGRAM_DATA"].reshape((num_packets, 24, 513))
    histogram_data = h[:, :, 1:]  # remove the pixel id field
    pixel_ids = h[:, :, 0]
    ts = TimeSeries(time=timestamps)
    ts["livetime"] = data["LIVE_TIME"]
    ts["inttime"] = data["INTEGRATION_TIME"]
    ts["pkttimes"] = data["TIME_S"]
    ts["pktclock"] = data["TIME_CLOCKS"]
    ts["seqcount"] = data["CCSDS_SEQUENCE_COUNT"]
    specs = Spectrum1D(
        spectral_axis=np.arange(histogram_data.shape[2]) * u.pix,
        flux=histogram_data * u.ct,
    )
    ts.meta.update({"ORIGFILE": f"{filename.name}"})

    # Clean Spectrum Times
    ts, specs, pixel_ids = clean_spectra_data(ts, specs, pixel_ids)

    return ts, specs, pixel_ids


def packet_definition_hist():
    """Return the packet definition for the histogram packets."""
    from ccsdspy import PacketArray, PacketField

    # NOTE: This is an outdated packet definition.
    # the number of pixels provided by a histogram packet
    NUM_BINS = 512
    NUM_PIXELS = 24

    # the header
    p = [
        PacketField(name="TIMESTAMPS", data_type="uint", bit_length=4 * 16),
        PacketField(name="TIMESTAMPCLOCK", data_type="uint", bit_length=4 * 16),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=32),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=32),
    ]

    for i in range(NUM_PIXELS):
        p += [
            PacketField(name=f"HISTOGRAM_SYNC{i}", data_type="uint", bit_length=8),
            PacketField(name=f"HISTOGRAM_DETNUM{i}", data_type="uint", bit_length=3),
            PacketField(name=f"HISTOGRAM_PIXNUM{i}", data_type="uint", bit_length=5),
            PacketArray(
                name=f"HISTOGRAM_DATA{i}",
                data_type="uint",
                bit_length=16,
                array_shape=NUM_BINS,
            ),
        ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_hist2():
    """Return the packet definition for the histogram packets."""
    # the number of pixels provided by a histogram packet
    NUM_BINS = 512
    NUM_PIXELS = 24
    from ccsdspy import PacketArray, PacketField

    # the header
    p = [
        PacketField(name="TIME_S", data_type="uint", bit_length=32),
        PacketField(name="TIME_CLOCKS", data_type="uint", bit_length=32),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=32),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=32),
    ]

    p += [
        PacketArray(
            name="HISTOGRAM_DATA",
            data_type="uint",
            bit_length=16,
            array_shape=NUM_PIXELS * (NUM_BINS + 1),
        ),
    ]

    p += [PacketField(name="CHECKSUM", data_type="uint", bit_length=16)]

    return p


def packet_definition_ph():
    """Return the packet definition for the photon packets."""
    from ccsdspy import PacketArray, PacketField

    p = [
        PacketField(name="TIME_S", data_type="uint", bit_length=32),
        PacketField(name="TIME_CLOCKS", data_type="uint", bit_length=32),
        PacketField(name="INTEGRATION_TIME", data_type="uint", bit_length=16),
        PacketField(name="LIVE_TIME", data_type="uint", bit_length=16),
        PacketField(name="FLAGS", data_type="uint", bit_length=16),
        PacketField(name="CHECKSUM", data_type="uint", bit_length=16),
        PacketArray(
            name="PIXEL_DATA", data_type="uint", bit_length=16, array_shape="expand"
        ),
    ]
    return p


def clean_spectra_data(
    ts: TimeSeries, spectra: Spectrum1D, ids: np.ndarray
) -> tuple[TimeSeries, Spectrum1D, np.ndarray]:
    """Given raw spectrum packet data, perform a cleaning operation that removes bad data.
    The most common cause of which are bits that are turned to zero when they should not be.

    This function finds unphysical times (before 2022-01-01) and replaces the time with an estimated time by using the median time between spectra.
    It also replaces all pixel ids with the median pixel id set.

    Parameters
    ----------
    ts : TimeSeries
        A TimeSeries object containing the time and other metadata.
    spectra : Spectrum1D
        A Spectrum1D object containing the spectral data.
    ids : np.ndarray
        An array of pixel ids corresponding to the spectra.

    Returns
    -------
    tuple[TimeSeries, Spectrum1D, np.ndarray]
        A tuple containing the cleaned TimeSeries, Spectrum1D, and pixel ids.
    """
    # Calculate Differences in Time-Related Columns
    dts = ts.time[1:] - ts.time[:-1]
    pkttimes_diff = ts["pkttimes"][1:] - ts["pkttimes"][:-1]
    pktclock_diff = ts["pktclock"][1:] - ts["pktclock"][:-1]

    # Calculate the Cadence to use for Interpolation
    median_dt = np.median(dts)
    pkttimes_diff_median = np.median(pkttimes_diff)
    pktclock_diff_median = np.median(pktclock_diff)

    bad_indices = np.argwhere(ts.time <= util.MIN_TIME_BAD)
    for this_bad_index in bad_indices:
        if this_bad_index < len(ts.time) - 1:
            ts.time[this_bad_index] = ts.time[this_bad_index + 1] - median_dt
            ts["pkttimes"][this_bad_index] = (
                ts["pkttimes"][this_bad_index + 1] - pkttimes_diff_median
            )
            ts["pktclock"][this_bad_index] = (
                ts["pktclock"][this_bad_index + 1] - pktclock_diff_median
            )
        else:
            ts.time[this_bad_index] = ts.time[this_bad_index - 1] + median_dt
            ts["pkttimes"][this_bad_index] = (
                ts["pkttimes"][this_bad_index - 1] + pkttimes_diff_median
            )
            ts["pktclock"][this_bad_index] = (
                ts["pktclock"][this_bad_index - 1] + pktclock_diff_median
            )

    # remove bad pixel ids
    median_ids = np.median(ids, axis=0)
    fixed_ids = (
        np.tile(median_ids, ids.shape[0])
        .reshape(ids.shape[0], len(median_ids))
        .astype(ids.dtype)
    )
    return ts, spectra, fixed_ids
