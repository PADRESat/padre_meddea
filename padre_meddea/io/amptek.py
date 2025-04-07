"""
This module provides io support for reading files created by the amptek detector.
"""

from pathlib import Path
from datetime import datetime

import numpy as np
from numpy.polynomial.polynomial import Polynomial

import astropy.units as u
from astropy.nddata import StdDevUncertainty
from astropy.io.fits import Header

from specutils import Spectrum1D
from specutils.spectra import SpectralRegion

from padre_meddea.util.util import str_to_fits_keyword


def read_mca(filename: Path, count_rate=False):
    """
    Read an amptek mca file.

    Parameters
    ----------
    filename: Path
        A file to read.
    count_rate: bool
        If True, then return data in count rate rather than counts.

    Returns
    -------
    spectrum: Spectrum1D
        All meta data with all capitals are direct from the mca file.
        Lower case metadata is added.

    Examples
    --------
    """
    with open(filename, "r", encoding="unicode_escape") as fp:
        in_data_section = False
        in_calib_section = False
        in_roi_section = False
        in_dp5config_section = False
        in_status_section = False
        in_spectrum_meta_section = False
        calib_data = {}
        roi_data = {}
        data = []
        line_number = 0
        hdr = Header()
        meta = {}
        for line in fp:
            #  figure out what section of the file we are in
            # the order used here is the order expected in the file
            # once in new section you are done with the previous section
            if line.count("PMCA SPECTRUM"):
                in_spectrum_meta_section = True
                continue
            if line.count("CALIBRATION"):
                in_calib_section = True
                in_spectrum_meta_section = False
                continue
            if line.count("ROI"):
                in_roi_section = True
                in_calib_section = False
                in_spectrum_meta_section = False
                continue
            if line.count("DATA"):
                in_data_section = True
                in_roi_section = False
                continue
            if line.count("END"):
                in_data_section = False
            if line.count("DP5 CONFIGURATION"):
                in_dp5config_section = True
                in_data_section = False
                continue
            if line.count("DPP STATUS"):
                in_dp5config_section = False
                in_status_section = True
                continue

            # now gather information
            if in_spectrum_meta_section:  # this is a measurement metadata tag
                keyword = line.split("-")[0]
                value = line.split("-")[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value))
            if in_calib_section:
                if line.count("-"):
                    continue
                else:
                    channel = float(line.split(" ")[0])
                    energy = float(line.split(" ")[1])
                    calib_data.update({channel: energy})
            if in_roi_section:
                roi_data.update({int(line.split(" ")[0]): int(line.split(" ")[1])})
            if in_data_section and not (line.count("<<") == 1):
                data.append(int(line))
            if in_dp5config_section and not (line.count("<<") == 1):
                keyword = line.split("=")[0].upper()
                value = line.split("=")[1].split(";")[0].strip()
                description = line.split("=")[1].split(";")[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value, description))
            if in_status_section and not (line.count("<<") == 1):
                keyword = line.split(":")[0]
                value = (
                    line.split(":")[1].strip().encode("ascii", "ignore").decode("ascii")
                )
                try:
                    value = float(value)
                except ValueError:
                    pass
                hdr.append((str_to_fits_keyword(keyword), value))
            line_number += 1
        if count_rate:
            y = u.Quantity(np.array(data) / hdr["REALTIME"], "ct/s")
            uncertainty = StdDevUncertainty(
                u.Quantity(np.sqrt(data) / hdr["REALTIME"], "ct/s")
            )
        else:
            y = data * u.ct
            uncertainty = StdDevUncertainty(np.sqrt(data) * u.ct)

        if roi_data:
            rois = SpectralRegion(
                [[x1 * u.pix, x2 * u.pix] for x1, x2 in roi_data.items()]
            )

        meta.update({"header": hdr})

        meta.update({"roi": rois})
        meta.update({"livetime": hdr["LIVETIME"] * u.s})
        meta.update({"realtime": hdr["REALTIME"] * u.s})
        meta.update(
            {"obs_time": datetime.strptime(hdr["starttim"], "%m/%d/%Y %H:%M:%S")}
        )
        meta.update({"filename": filename})
        meta.update({"dtimfrac": 1 - hdr["livetime"] / hdr["realtime"]})
        meta.update({"rate": np.sum(data) * u.ct / meta["realtime"]})
        if len(calib_data) > 0:
            channels = list(calib_data.keys())
            energies = list(calib_data.values())
            meta.update({"calib": Polynomial.fit(channels, energies, deg=1)})

        spectrum = Spectrum1D(
            flux=y,
            spectral_axis=np.arange(len(data)) * u.pix,
            uncertainty=uncertainty,  # TODO add uncertainty, problem with ct/s as unit
            meta=meta,
        )

    return spectrum
