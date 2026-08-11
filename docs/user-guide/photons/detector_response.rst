.. _drm:

******************************
Detector Response Matrix (DRM)
******************************

The detector response matrix (DRM) describes the response of a detector to incoming photons. It is used to convert the incident photon spectrum into the observed count spectrum.
The response of a detector is typically characterized by two components: the ancillary response file (ARF) and the redistribution matrix file (RMF). The ARF describes the effective area of the detector as a function of energy, while the RMF describes how photons of different energies are redistributed into different detector channels.
The DRM is typically represented as a two-dimensional matrix, where the rows correspond to the incident photon energies and the columns correspond to the detector channels. Each element of the matrix represents the probability that a photon of a given energy will be detected in a particular channel.

To get the latest DRM files, use the `get_drm_files` function from the `padre_meddea.spectrum.calibration` module. This function returns a list of paths to the ARF and RMF files.

.. doctest::

    >>> from padre_meddea.spectrum.calibration import get_drm_files
    >>> rmf_file, arf_file = get_drm_files()
    >>> print(arf_file.name)
    20260721_meddea_arf.fits
    >>> print(rmf_file.name)
    20260721_meddea_rmf.fits

They are stored using the OGIP format, which is a standard format for storing DRM files in the FITS file format.
The OGIP format is widely used in the high-energy astrophysics community and is supported by many software packages.

.. plot::

    import matplotlib.pyplot as plt
    from padre_meddea.spectrum.calibration import get_drm_files
    from astropy.io import fits

    rmf_file, arf_file = get_drm_files()
    with fits.open(arf_file) as hdul:
        arf = hdul[1].data

        energ_lo = arf["ENERG_LO"]
        energ_hi = arf["ENERG_HI"]
        specresp = arf["SPECRESP"]

    # Define the energy bin centers.
    energy_center = (energ_lo + energ_hi)/2.

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(energy_center, specresp, lw=1.5)

    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Effective Area (cm$^2$)")
    ax.set_title("MeDDEA Ancillary Response File (ARF)")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


.. plot::

    import matplotlib.pyplot as plt
    from padre_meddea.spectrum.calibration import get_drm_files
    from astropy.io import fits
    import numpy as np
    from matplotlib import colors

    rmf_file, arf_file = get_drm_files()

    with fits.open(rmf_file) as hdul:

        matrix_hdu = hdul["MATRIX"]
        ebounds_hdu = hdul["EBOUNDS"]

        matrix_data = matrix_hdu.data
        ebounds_data = ebounds_hdu.data

        # Incident photon energy bins.
        energ_lo = matrix_data["ENERG_LO"]
        energ_hi = matrix_data["ENERG_HI"]

        # Measured photon energy bins.
        e_min = ebounds_data["E_MIN"]
        e_max = ebounds_data["E_MAX"]

        n_energy = len(energ_lo)
        n_channels = len(e_min)

        # Reconstruct the full RMF matrix.
        rmf_matrix = np.zeros((n_energy, n_channels), dtype=float)

        for i in range(n_energy):

            n_grp = matrix_data["N_GRP"][i]

            f_chan = matrix_data["F_CHAN"][i]
            n_chan = matrix_data["N_CHAN"][i]
            values = matrix_data["MATRIX"][i]

            idx = 0

            for g in range(n_grp):
                start = f_chan[g]
                length = n_chan[g]

                rmf_matrix[i, start:start+length] = values[idx:idx+length]

                idx += length


    # Energy centers for plotting
    true_energy = (energ_lo + energ_hi)/2.
    measured_energy = (e_min + e_max)/2.


    # Avoid accidentally calculating log(0).
    positive_values = rmf_matrix[rmf_matrix > 0]
    vmin = positive_values.min()
    vmax = positive_values.max()

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(8, 6))

    extent = [
        measured_energy.min(),
        measured_energy.max(),
        true_energy.min(),
        true_energy.max()
    ]

    im = ax.imshow(
        rmf_matrix,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        norm=norm
    )

    ax.set_xlabel("Measured Energy (keV)")
    ax.set_ylabel("Incident Photon Energy (keV)")
    ax.set_title("MeDDEA Response Matrix File (RMF)")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Redistribution Probability")

    plt.tight_layout()
    plt.show()
