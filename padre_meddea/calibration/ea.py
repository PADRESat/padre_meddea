""""""

import astropy.units as u
from astropy.table import QTable
from roentgen.absorption import Material


def get_effective_area(energy: u.keV, no_al=False, no_be=False, no_mli=False):
    """Returns the model effective area

    Parameters
    ----------
    energy : u.keV
    no_al : bool, optional
        If True, ignore the aluminum layer. Default is False.
    no_be : bool, optional
        If True, ignore the beryllium layer. Default is False.
    no_mli : bool, optional
        If True, ignore the MLI blankets. Default is False.
    Returns
    -------
    effective_area : u.cm^2
    """

    al_thickness = 0.4 * u.mm
    be_thickness = 1 * u.mm
    number_of_detectors = 4
    # caliste so properties see https://github.com/i4Ds/STIXCore/blob/master/stixcore/calibration/transmission.py#L18
    geo_area_detector = 0.81 * u.cm**2
    detector_thickness = 1 * u.mm
    meddea_geo_area = number_of_detectors * geo_area_detector

    det_cathode = Material("Pt", 30 * u.nm)
    detector_efficiency = Material("cdte", thickness=detector_thickness).absorption(
        energy
    ) * det_cathode.transmission(energy)
    # MLI blankets
    MIL_SI = 0.0254 * u.mm

    blankets = Material("C", (3 + 20 * 0.05 + 3) * MIL_SI)

    detector_area = meddea_geo_area * detector_efficiency
    effective_area = detector_area

    if not no_mli:
        effective_area *= blankets.transmission(energy)
    if not no_al:
        effective_area *= Material("Al", thickness=al_thickness).transmission(energy)
    if not no_be:
        effective_area *= Material("Be", thickness=be_thickness).transmission(energy)
    return effective_area


def get_ea_table(energy_ax: u.keV, num_detectors=1):
    base_ea = get_effective_area(energy_ax) / 4.0 * num_detectors
    no_al_ea = get_effective_area(energy_ax, no_al=True) / 4.0 * num_detectors
    no_filters = (
        get_effective_area(energy_ax, no_al=True, no_be=True) / 4.0 * num_detectors
    )

    result = QTable()
    result["energy"] = energy_ax
    result["base"] = base_ea
    result["no_al"] = no_al_ea
    result["no_filters"] = no_filters

    return result
