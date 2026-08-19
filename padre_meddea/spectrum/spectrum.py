"""
Provides data containers for both summary spectra and event list or photon data.
"""

import astropy.units as u
import numpy as np
from astropy.nddata import StdDevUncertainty
from astropy.table import Table, vstack
from astropy.time import Time
from astropy.timeseries import BinnedTimeSeries, TimeSeries, aggregate_downsample
from specutils import SpectralRegion, Spectrum1D

from padre_meddea.util.pixels import PixelList, get_pixelid

DEFAULT_SPEC_PIXEL_IDS = np.array(
    [
        51738,
        51720,
        51730,
        51712,
        51733,
        51715,
        51770,
        51752,
        51762,
        51744,
        51765,
        51747,
        51802,
        51784,
        51794,
        51776,
        51797,
        51779,
        51834,
        51816,
        51826,
        51808,
        51829,
        51811,
    ],
    dtype=np.uint16,
)
MAX_PH_DATA_RATE = 100 * u.kilobyte / u.s

DEFAULT_SPEC_PIXEL_LIST = PixelList(pixelids=DEFAULT_SPEC_PIXEL_IDS)

__all__ = [
    "PhotonList",
    "SpectrumList",
]


class PhotonList:
    """Data container for MeDDEA photon or event list data

    Parameters
    ----------
    pkt_list : TimeSeries
        The time series of photon packet header data.
    event_list : TimeSeries
        The time series of event data

    TODO: add metadata if opening from fits file

    Examples
    --------
    >>> from padre_meddea.io import read_file
    >>> from padre_meddea.util.pixels import PixelList
    >>> ph_list = read_file("padre_meddea_l0test_photons_20250504T070411_v0.1.0.fits")  # doctest: +SKIP
    >>> this_spectrum = ph_list.spectrum(pixel_list=ph_list.pixel_list)  # doctest: +SKIP
    """

    def __init__(
        self, pkt_list: TimeSeries, event_list: TimeSeries, meta: dict | None = None
    ):
        self.data = {"event_list": event_list, "pkt_list": pkt_list}
        self.event_list = event_list
        self.pkt_list = pkt_list
        self.time = self.event_list.time
        self.meta = meta

    def __getitem__(self, key):
        if isinstance(key, int):
            raise ValueError(
                "PhotonList does not support indexing by integer. Use a slice with start and stop times instead."
            )
        if isinstance(key, str):
            raise ValueError(
                "PhotonList does not support indexing by string. Use a slice with start and stop times instead."
            )
        elif isinstance(key, slice):
            if isinstance(key.start, int) or isinstance(key.stop, int):
                raise ValueError(
                    "PhotonList does not support indexing by integer. Use a slice with start and stop times instead."
                )
            if isinstance(key.start, str) and isinstance(key.stop, str):
                start_time = Time(key.start)
                stop_time = Time(key.stop)
            elif isinstance(key.start, Time) and isinstance(key.stop, Time):
                start_time = key.start
                stop_time = key.stop
            else:
                raise ValueError(
                    f"Invalid slice type, {type(key.start)} and {type(key.stop)}. Must be str or Time."
                )
            pkt_ind = (self.pkt_list.time >= start_time) * (
                self.pkt_list.time <= stop_time
            )
            ph_ind = (self.event_list.time >= start_time) * (
                self.event_list.time <= stop_time
            )
            if len(pkt_ind) == 0 or len(ph_ind) == 0:
                raise ValueError(f"No data found between {start_time} and {stop_time}")
            else:
                return type(self)(self.pkt_list[pkt_ind], self.event_list[ph_ind])

        return self

    def __str__(self):
        return f"{self._text_summary()}{self.data.__repr__()}"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    def _text_summary(self):
        num_events = len(self.data["event_list"])
        result = f"PhotonList ({num_events:,} events)\n"
        if num_events > 0:
            dt = self.data["event_list"].time[-1] - self.data["event_list"].time[0]
            dt.format = "quantity_str"
            if dt < (1 * u.day):
                result += f"{self.data['event_list'].time[0]} - {str(self.data['event_list'].time[-1])[11:]} ({dt})\n"
            else:
                result += f"{self.data['event_list'].time[0]} - {self.data['event_list'].time[-1]} ({dt})\n"
        return result

    def __add__(self, other: PixelList):
        event_list = vstack([self.event_list, other.event_list])
        pkt_list = vstack([self.pkt_list, other.pkt_list])
        return type(self)(pkt_list, event_list)

    def calibrate(self):
        """Calibrate atod values to energy units.

        Code provided by Muriel Stiefel (FHNW)"""

        # gain and offset values for each pixel from Olivier Limousin

        # TODO: these values are for ground calibration, need to update for flight calibration
        # TODO: calibration should not be hard coded, should be read from a calibration file and should vary as a function of time.

        # fmt: off
        gain_ground = np.array([7.955967, 7.932099, 7.895062, 7.9316874, 7.8909464, 7.869959,
                                7.8884773, 7.916461, 7.851029, 7.9720163, 7.9074078, 7.911523,
                                8.015638, 7.9798355, 7.955967, 7.9798355, 8.031687, 7.960082,
                                7.8925924, 7.936214, 7.911523, 8.027983, 7.8831277, 7.973663,
                                8.056378, 8.08107, 8.060494, 8.095884, 8.096708, 8.007819,
                                8.098765, 8.079425, 8.165431, 8.068724, 8.100823, 8.126339,
                                8.137038, 8.166666, 8.14568, 8.189301, 8.224691, 8.067902,
                                8.155555, 8.235391, 8.203704, 8.197119, 8.14321, 8.179012])

        offset_ground = np.array([39.34156, 38.674896, 45.065845, 40.530865, 48.19753, 44.732513,
                            44.79424, 43.798355, 32.440327, 49.85597, 43.68313, 41.954735,
                            54.736626, 57.23045, 73.37449, 52.831276, 49.436214, 55.64609,
                            68.148155, 56.390945, 55.349792, 62.930042, 56.045265, 60.522636,
                            42.44856, 40.27572, 36.514404, 47.024693, 29.596708, 34.958847,
                            50.975307, 33.152264, 36.23868, 37.65844, 26.255144, 35.148148,
                            59.93827, 55.01646, 59.666668, 47.312756, 43.85597, 63.68313,
                            51.36214, 48.88066, 57.045265, 50.074074, 61.62963, 54.679012])
        # fmt: on

        gain_flight = gain_ground * 4
        offset_flight = offset_ground * 4

        if "baseline" not in self.event_list.colnames:
            raise ValueError(
                "Baseline column not found in event list. Cannot calibrate data."
            )

        atod = (
            self.event_list["atod"].astype(float)
            - self.event_list["baseline"].astype(float)
        ) + 3572.86
        pixel_index = self.event_list["asic"] * 12 + self.event_list["pixel"]

        self.event_list["energy"] = (atod - offset_flight[pixel_index]) / gain_flight[
            pixel_index
        ]

    @property
    def calibrated(self):
        if "energy" in self.event_list.colnames:
            return True
        else:
            return False

    @property
    def pixel_list(self) -> PixelList:
        """Return the set of pixels that have events"""
        # note this is calculated on the fly instead of at init because it can take a few seconds to compute for large event lists
        pixel_ids = np.unique(
            get_pixelid(self.event_list["asic"], self.event_list["pixel"])
        )
        return PixelList(pixelids=pixel_ids)

    def spectrum(
        self,
        pixel_list: PixelList | None = None,
        bins: np.ndarray | None = None,
    ) -> Spectrum1D:
        """
        Create a spectrum

        Parameters
        ----------
        pixel_list : PixelList
            A list of one or more pixels, if None, then uses all pixels
        bins : np.array
            The bin edges for the spectrum (see ~np.histogram).
            If None and the spectrum is not calibrated, then uses np.arange(0, 2**12 - 1) * u.pix,
            If None and the spectrum is calibrated then uses np.arange(3, 100, 0.1) * u.keV

        Returns
        -------
        spectrum : Spectrum1D
        """
        if pixel_list:
            this_event_list = self._slice_event_list_pixels(pixel_list)
        else:
            this_event_list = self.event_list

        if self.calibrated:
            bins = np.arange(3, 100, 0.1) * u.keV
            hit_energy = this_event_list["energy"]
        else:
            bins = np.arange(0, 2**12 - 1) * u.pix
            hit_energy = this_event_list["atod"]

        data, new_bins = np.histogram(hit_energy, bins=bins.value)

        # for Spectrum1D, the spectral axis is at the center of the bins
        # TODO: the histogram results are not consistent with the above
        result = Spectrum1D(
            flux=u.Quantity(data, "count"),
            spectral_axis=bins,
            uncertainty=StdDevUncertainty(np.sqrt(data) * u.count),
        )
        return result

    def lightcurve(
        self,
        pixel_list: PixelList | None = None,
        time_bin_size: u.Quantity | None = 1 * u.s,
        energy_edges: u.Quantity | None = None,
    ) -> TimeSeries:
        """
        Create a light curve

        Parameters
        ----------
        pixel_list : PixelList
            The pixels to integrate over
        int_time : u.Quantity[u.s]
            The integration time for each time step
        sr : SpectralRegion
            The spectral region(s) to integrate over
        step : int
            To speed up processing, skip every `step` photons.
            Default is ten.
            The light curve count rate is corrected by multiplying by `step`.

        Returns
        -------
        lc : TimeSeries
        """
        if not self.calibrated:
            raise ValueError(
                "Data must be calibrated before plotting a spectrogram. Call the `calibrate()` method first."
            )
        if pixel_list is None:
            energies_to_plot = self.event_list["energy"] * u.keV
        else:
            energies_to_plot = (
                self._slice_event_list_pixels(pixel_list)["energy"] * u.keV
            )
        if energy_edges is None:
            energy_min = np.min(energies_to_plot)
            energy_max = np.max(energies_to_plot)
            energy_edges = [energy_min, energy_max]
            times_to_plot = self.event_list.time
        else:
            energy_min = np.min(energy_edges)
            energy_max = np.max(energy_edges)
            mask = (energies_to_plot >= energy_min) * (energies_to_plot <= energy_max)
            energies_to_plot = energies_to_plot[mask]
            times_to_plot = self.event_list.time[mask]

        if len(times_to_plot) == 0:
            raise ValueError("No events available after applying the energy filter")

        tstart = times_to_plot[0]
        tend = times_to_plot[-1]
        time_edges = np.arange(
            tstart.unix,
            tend.unix + time_bin_size.to(u.s).value,
            time_bin_size.to(u.s).value,
        )
        if len(time_edges) < 2:
            time_edges = np.array(
                [tstart.unix, tend.unix + time_bin_size.to(u.s).value]
            )

        hist, _, _ = np.histogram2d(
            times_to_plot.unix,
            energies_to_plot.to_value(u.keV),
            bins=[time_edges, energy_edges.to_value(u.keV)],
        )
        ts = TimeSeries(time=Time(time_edges[:-1], format="unix"))
        ts.time.format = "isot"
        for i, this_range in enumerate(energy_edges[:-1]):
            col_label = f"{energy_edges[i].value:0.0f}to{energy_edges[i + 1].value:0.0f}_{energy_edges[i + 1].unit}"
            ts[col_label] = hist[:, i]
        return ts

    def plot_spectrogram(
        self,
        energy_range: u.Quantity | None = None,
        time_bin_size: u.Quantity | None = 1 * u.s,
        energy_bin_size: u.Quantity | None = 0.1 * u.keV,
        log_color: bool = True,
        **imshow_kwargs,
    ):
        """
        Plot a 2D spectrogram of photon events with time on the x-axis and energy on the y-axis.
        Note, data must be calibrated first.

        Parameters
        ----------
        energy_range : u.Quantity, optional
            The energy range to plot. If None, uses the full range of the data.
        time_bin_size : u.Quantity, optional
            The time bin size for the spectrogram. Default is 1 second.
            If None, uses the default value of 1 second.
        energy_bin_size : u.Quantity, optional
            The energy bin size for the spectrogram. Default is 0.1 keV.
        log_color : bool, optional
            Whether to use a logarithmic color scale. Default is True.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        import matplotlib.colors as mcolors
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if not self.calibrated:
            raise ValueError(
                "Data must be calibrated before plotting a spectrogram. Call the `calibrate()` method first."
            )
        energies_to_plot = self.event_list["energy"] * u.keV

        if energy_range is None:
            energy_min = energies_to_plot.min()
            energy_max = energies_to_plot.max()
            times_to_plot = self.event_list.time
        else:
            energy_min, energy_max = energy_range
            mask = (energies_to_plot >= energy_min) * (energies_to_plot <= energy_max)
            energies_to_plot = energies_to_plot[mask]
            times_to_plot = self.event_list.time[mask]

        if len(times_to_plot) == 0:
            raise ValueError("No events available after applying the energy filter")

        tstart = times_to_plot[0]
        tend = times_to_plot[-1]
        time_edges = np.arange(
            tstart.unix,
            tend.unix + time_bin_size.to(u.s).value,
            time_bin_size.to(u.s).value,
        )
        if len(time_edges) < 2:
            time_edges = np.array(
                [tstart.unix, tend.unix + time_bin_size.to(u.s).value]
            )

        energy_edges = (
            np.arange(
                energy_min.value,
                energy_max.value + energy_bin_size.to(u.keV).value,
                energy_bin_size.to(u.keV).value,
            )
            * u.keV
        )
        if len(energy_edges) < 2:
            energy_edges = np.array([energy_min, energy_max])

        hist, _, _ = np.histogram2d(
            times_to_plot.unix,
            energies_to_plot.to_value(u.keV),
            bins=[time_edges, energy_edges.to_value(u.keV)],
        )

        fig, ax = plt.subplots()

        if log_color:
            hist_plot = np.where(hist > 0, hist, np.nan)
            vmax = max(1, np.nanmax(hist_plot))
            norm = mcolors.LogNorm(vmin=1, vmax=vmax)
        else:
            hist_plot = hist
            norm = None

        time_dt = [Time(t, format="unix").to_datetime() for t in time_edges]
        time_num = mdates.date2num(time_dt)

        image = ax.pcolormesh(
            time_num,
            energy_edges.to_value(u.keV),
            hist_plot.T,
            cmap="viridis",
            norm=norm,
            shading="auto",
        )
        plt.colorbar(image, ax=ax, label="Counts")

        ax.set_xlabel(f"Time {tstart.iso} to {tend.iso}")
        ax.set_ylabel("Energy [keV]")
        ax.set_title(f"MeDDEA spectrogram (Δt={time_bin_size}, ΔE={energy_bin_size})")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
        ax.set_xlim(time_num[0], time_num[-1])

        fig.autofmt_xdate()

        return fig, ax

    def data_rate(self) -> BinnedTimeSeries:
        """Return a BinnedTimeseries of the data rate.

        Returns
        -------
        data_rate : BinnedTimeSeries
        """
        # correct the ccsds packet length by adding ccsds header and adding missing 1
        pkt_length = (self.pkt_list["pktlength"] + 3 * 2 + 1) * u.byte
        good_times = (
            self.pkt_list.time > self.pkt_list.time[0]
        )  # to protect against bad times

        data_rate = TimeSeries(
            time=self.pkt_list.time[good_times],
            data={"packet_size": pkt_length[good_times]},
        )
        data_rate_ts = aggregate_downsample(
            data_rate, time_bin_size=1 * u.s, aggregate_func=np.sum
        )
        data_rate_ts.rename_column("packet_size", "data_rate")
        data_rate_ts["data_rate"] = data_rate_ts["data_rate"] / u.s
        return data_rate_ts

    def _slice_event_list_pixels(self, pixel_list: PixelList) -> TimeSeries:
        """Slice the event list to only contain events from asic_num and pixel_num"""
        ind = np.zeros(len(self.event_list))
        if isinstance(pixel_list, Table.Row):
            ind = np.logical_or(
                ind,
                (self.event_list["pixel"] == int(pixel_list["pixel"]))
                * (self.event_list["asic"] == int(pixel_list["asic"])),
            )
        else:
            for this_pixel in pixel_list:
                ind = np.logical_or(
                    ind,
                    (self.event_list["pixel"] == int(this_pixel["pixel"]))
                    * (self.event_list["asic"] == int(this_pixel["asic"])),
                )
        return self.event_list[ind]

    def _slice_event_list_sr(self, sr: SpectralRegion):
        """Slice the envt list to only contain events inside the spectral region."""
        if len(sr) > 1:
            raise ValueError("Only supports Spectral Regions of length 1.")
        if sr[0].lower.unit == u.Unit("keV"):
            data = self.event_list["energy"] * u.pix
        elif sr[0].lower.unit == u.Unit("pix"):
            data = self.event_list["atod"] * u.pix
        else:
            raise ValueError(
                f"Unit of Spectral Region, {sr[0].lower.unit}, not recognized."
            )
        ind = (data > sr[0].lower) * (data < sr[0].upper)
        return self.event_list[ind]


class SpectrumList:
    """
    A data container for MeDDEA summary spectrum data

    Parameters
    ----------
    pkt_spec : TimeSeries
        The time series of spectrum packet header data.
    specs : Spectrum1D
        The spectrum cube
    pixel_ids : np.array
        The pixel id array

    Raises
    ------
    ValueError
        If pixel arrays are found to change.

    Examples
    --------
    >>> from padre_meddea.io import read_file
    >>> from astropy.time import Time
    >>> spec_list = read_file("padre_meddea_l0test_spectrum_20250504T070411_v0.1.0.fits")  # doctest: +SKIP
    >>> this_spectrum = spec_list.spectrum(pixel_list=spec_list.pixel_list)  # doctest: +SKIP
    """

    def __init__(self, pkt_list: TimeSeries, specs, pixel_ids):
        self.bins = np.arange(0, 4097, 8, dtype=np.uint16)
        self.time = pkt_list.time
        self.data = {"pkt_list": pkt_list, "specs": specs, "pixel_ids": pixel_ids}
        self.pkt_list = self.data["pkt_list"]
        self.specs = self.data["specs"]
        self._pixel_ids = self.data["pixel_ids"]
        if len(np.unique(pixel_ids)) > 24:
            print("Found too many unique pixel IDs.")
            print("Forcing to default set")
            self.pixel_list = PixelList(pixelids=DEFAULT_SPEC_PIXEL_IDS)
        else:
            if np.all(np.unique(pixel_ids) == sorted(pixel_ids[0, :])):
                self.pixel_list = PixelList(
                    pixelids=np.median(pixel_ids, axis=0).astype("uint16")
                )
            else:
                raise ValueError("Found change in pixel ids")
        self.index = len(pkt_list)

    @property
    def calibrated(self):
        if self.specs[0, 0].spectral_axis.unit == u.Unit("keV"):
            return True
        else:
            return False

    def __str__(self):
        return f"{self._text_summary()}{self.data['specs'].__repr__()}"

    def __repr__(self):
        return f"{object.__repr__(self)}\n{self}"

    def _text_summary(self):
        dt = self.time[-1] - self.time[0]
        dt.format = "quantity_str"
        result = f"SpectrumList ({self.specs.shape[0]:,} spectra, {int(np.sum(self.specs.data)):,} events)\n"
        if dt < (1 * u.day):
            result += f"{self.time[0]} - {str(self.time[-1])[11:]} ({dt})\n"
        else:
            result += f"{self.time[0]} - {self.time[-1]} ({dt})\n"
        return result

    def spectrum(self, pixel_list: PixelList):
        """Create a spectrum, integrates over all times

        Parameters
        ----------
        asic_num : int
            The asic or detector number (0 to 3)
        pixel_num : int
            The pixel number (0 to 11)
        or
        spec_index : int
            The spectrum index from 0 to 23

        Raises
        ------
        ValueError
            If the selected asic_num and pixel_num are not found in the spectra

        Returns
        -------
        spectrum : Spectrum1D
        """
        flux = np.zeros([self.specs.data.shape[2]])
        if isinstance(pixel_list, Table.Row):
            if pixel_list in self.pixel_list:
                pixel_index = np.where(pixel_list == self.pixel_list)[0][0]
                flux += np.sum(self.specs.data[:, pixel_index, :], axis=0)
        else:
            for this_pixel in pixel_list:
                if this_pixel in self.pixel_list:
                    pixel_index = np.where(this_pixel == self.pixel_list)[0][0]
                    flux += np.sum(self.specs.data[:, pixel_index, :], axis=0)
        # the spectral axis is at the center of the bins
        result = Spectrum1D(
            flux=flux * self.specs[0, 0].flux.unit,
            spectral_axis=self.specs[0, 0].spectral_axis,
            uncertainty=StdDevUncertainty(np.sqrt(flux) * u.count),
        )
        return result

    def lightcurve(self, pixel_list: PixelList, sr: SpectralRegion) -> TimeSeries:
        """
        Create a light curve

        Parameters
        ----------
        pixel_index : int
            The pixels to integrate over
        sr : SpectralRegion
            The spectral region(s) to integrate over

        Returns
        -------
        lc : TimeSeries
        """
        lc = TimeSeries(time=self.time)
        flux = np.zeros([self.specs.data.shape[0], self.specs.data.shape[2]])
        if isinstance(pixel_list, Table.Row):
            if pixel_list in self.pixel_list:
                pixel_index = np.where(pixel_list == self.pixel_list)[0][0]
                flux += self.specs.data[:, pixel_index, :]
        else:
            for this_pixel in pixel_list:
                if this_pixel in self.pixel_list:
                    pixel_index = np.where(this_pixel == self.pixel_list)[0][0]
                    flux += self.specs.data[:, pixel_index, :]
        for i, this_sr in enumerate(sr):
            if str(self.specs.spectral_axis.unit) != str(this_sr.lower.unit):
                raise ValueError(
                    f"Units of spectral axis ({self.specs.spectral_axis.unit}) does not match units of sr ({this_sr.lower.unit})"
                )
            this_flux = flux.copy()
            ind = (self.specs.spectral_axis > this_sr.lower) * (
                self.specs.spectral_axis < this_sr.upper
            )
            this_flux[:, ~ind] = 0
            col_label = f"{this_sr.lower.value:0.0f}to{this_sr.upper.value:0.0f}_{this_sr.upper.unit}"
            total_cts = np.sum(this_flux, axis=1)
            lc[col_label] = total_cts
        return lc

    def spectrogram(self):
        specgram = np.sum(self.specs.data, axis=1) * u.ct
        ts = TimeSeries(time=self.time, data={"specgram": specgram})
        ts.meta["spectral_axis"] = u.Quantity(self.specs.spectral_axis)
        return ts

    def plot_spectrogram(self, **imshow_kwargs):
        """Plot a spectrogram"""
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        ts = [mdates.date2num(this_time) for this_time in self.time.to_datetime()]
        x_lims = [ts[0], ts[-1]]
        y_lims = [
            self.specs[0, 0].spectral_axis[0].value,
            self.specs[0, 0].spectral_axis[-1].value,
        ]
        fig, ax = plt.subplots()
        # TODO use spectrogram function above
        specgram = np.sum(self.specs.data, axis=1)
        ax.imshow(
            specgram.transpose(),
            origin="lower",
            interpolation="nearest",
            extent=[x_lims[0], x_lims[1], y_lims[0], y_lims[1]],
            **imshow_kwargs,
        )
        date_format = mdates.DateFormatter("%H:%M:%S")
        ax.xaxis.set_major_formatter(date_format)
        # This simply sets the x-axis data to diagonal so it fits better.
        fig.autofmt_xdate()
        plt.show()

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.specs[key]
        elif isinstance(key, slice):
            if isinstance(key.start, Time) and isinstance(key.stop, Time):
                ind = (self.time > key.start) * (self.time < key.stop)
                return type(self)(
                    self.pkt_list[ind], self.specs[ind, :, :], self._pixel_ids
                )
        return self
