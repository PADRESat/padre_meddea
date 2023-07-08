.. _logger:

**************
Logging system
**************

Overview
========

The logging system is an adapted version of `~astropy.logger.AstropyLogger`.
Its purpose is to provide users the ability to decide which log and warning messages to show,
to capture them, and to send them to a file.

All messages use this logging facility which is based
on the Python `logging` module rather than print statements
which will save the messages to a local file called ``padre_meddea.log``.