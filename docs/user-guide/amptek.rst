.. _amptek:

*************************
Amptek Reference Detector
*************************

Overview
========

For many ground-based calibration tasks, an amptek detector was used.
To facilitate the analysis of these data, this package includes some support for opening and analyzing these calibration files.

A test file is included of a measurement of a Mini-X xray tube with a gold target.

.. plot::

    An amptek Mini-X xray tube spectrum.

    >>> import numpy as np
    >>> import padre_meddea
    >>> from padre_meddea.io.amptek import read_mca
    >>> mca_file = padre_meddea._test_files_directory / "minix_20kV_15uA_sdd.mca"
    >>> spec = read_mca(mca_file)
    >>> spec.plot()


The output of `~read_mca` is a Spectrum object.
It contains a bunch of information inside its metadata which is a dictionary.
For example, if the mca file contains calibration data, a calibration function is provided.
And if regions of interest were defined those are also provided.

.. plot::

    An amptek Mini-X xray tube spectrum.

    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import astropy.units as u
    >>> import padre_meddea
    >>> from padre_meddea.io.amptek import read_mca
    >>> mca_file = padre_meddea._test_files_directory / "minix_20kV_15uA_sdd.mca"
    >>> spec = read_mca(mca_file)
    >>> f = spec.meta['calib']
    >>> energy_ax = f(spec.spectral_axis.value)
    >>> fig, ax = plt.subplots(layout="constrained")
    >>> ax.plot(energy_ax, spec.flux)
    >>> ax.set_xlabel("energy [keV]")
    >>> ax.set_ylabel("Counts")
    >>> for this_roi in spec.meta['roi']:
    >>>     ax.axvspan(f(this_roi.lower.value), f(this_roi.upper.value), alpha=0.2)

The regions of interest are provided as SpectralRegion objects.
They can be used to easily extract a sub spectrum to, for example, fit the gaussian to the emission line.

.. plot::

    An amptek Mini-X xray tube spectrum.

    >>> import matplotlib.pyplot as plt
    >>> import padre_meddea
    >>> from padre_meddea.io.amptek import read_mca
    >>> from specutils.manipulation import extract_region
    >>> from specutils.fitting import estimate_line_parameters, fit_lines
    >>> from astropy.modeling import models
    >>> mca_file = padre_meddea._test_files_directory / "minix_20kV_15uA_sdd.mca"
    >>> spec = read_mca(mca_file)
    >>> this_roi = spec.meta['roi'][0]
    >>> sub_spec = extract_region(spec, this_roi)
    >>> params = estimate_line_parameters(sub_spec, models.Gaussian1D())
    >>> g_init = models.Gaussian1D(amplitude=params.amplitude, mean=params.mean, stddev=params.stddev)
    >>> g_fit = fit_lines(sub_spec, g_init)
    >>> fig, ax = plt.subplots(layout="constrained")
    >>> ax.plot(sub_spec.spectral_axis, sub_spec.flux)
    >>> ax.plot(sub_spec.spectral_axis, g_fit(sub_spec.spectral_axis), label=f"{g_fit}")
    >>> plt.legend(bbox_to_anchor =(0.65, 1.25))
