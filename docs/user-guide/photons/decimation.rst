.. _decimation:

**********
Decimation
**********

In the ideal case, every event in the Caliste-SO detectors is recorded.
At high rates, a decimation mechanism is applied to ensure that high energy events, which are less frequent than low energy events, are prioritized.
Decimation is defined by two quantities, the energy range and the fraction of events discarded.
They are stored in decimation table files.
Since the decimation configuration can be changed over time, each file is associated with the start date for when it applies.


.. decimation-table-file:

A sample decimation configuration table
---------------------------------------

.. only:: html

.. literalinclude:: ../../../padre_meddea/data/decimation/20250314_decimation_table.csv

Keeps are the number of events that are recorded while discards are the numbers discarded.
The fraction of events discarded can be calculated by (discards) / (keeps + discards).
If keeps is 1 and discard is 2 then one event is recorded then the next two events are discarded.
The fraction of events discarded is then (2) / (2 + 1) of 2/3.

There are 7 decimation levels.
The decimation level is set by the event rate.
The higher the rate the higher the decimation level and the higher the discarded event ratio.
Generally, as the decimation level increases by 1 the ratio of discarded events increases by a factor of 2.