"""
This module provides errors/exceptions and warnings of general use.

Exceptions that are specific to a given package should **not** be here,
but rather in the particular package.

This code is based on that provided by SunPy see
    licenses/SUNPY.rst
"""

import warnings

__all__ = [
    "MEDDEAWarning",
    "MEDDEAUserWarning",
    "MEDDEADeprecationWarning",
    "MEDDEAPendingDeprecationWarning",
    "warn_user",
    "warn_deprecated",
]


class MEDDEAWarning(Warning):
    """
    The base warning class from which all PADRE MEDDEA warnings should inherit.

    Any warning inheriting from this class is handled by the PADRE MEDDEA
    logger. This warning should not be issued in normal code. Use
    "MEDDEAUserWarning" instead or a specific sub-class.
    """


class MEDDEAUserWarning(UserWarning, MEDDEAWarning):
    """
    The primary warning class for PADRE MEDDEA.

    Use this if you do not need a specific type of warning.
    """


class MEDDEADeprecationWarning(FutureWarning, MEDDEAWarning):
    """
    A warning class to indicate a deprecated feature.
    """


class MEDDEAPendingDeprecationWarning(PendingDeprecationWarning, MEDDEAWarning):
    """
    A warning class to indicate a soon-to-be deprecated feature.
    """


def warn_user(msg, stacklevel=1):
    """
    Raise a `MEDDEAUserWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, MEDDEAUserWarning, stacklevel + 1)


def warn_deprecated(msg, stacklevel=1):
    """
    Raise a `MEDDEADeprecationWarning`.

    Parameters
    ----------
    msg : str
        Warning message.
    stacklevel : int
        This is interpreted relative to the call to this function,
        e.g. ``stacklevel=1`` (the default) sets the stack level in the
        code that calls this function.
    """
    warnings.warn(msg, MEDDEADeprecationWarning, stacklevel + 1)
