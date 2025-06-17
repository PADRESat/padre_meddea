import astropy.units as u


def pulser_frequency(interval: int) -> u.Quantity:
    """Return the frequency of the pulser output given the interval setting."""
    return 20000000 / ((interval * 65536) + 20000) * u.Hz