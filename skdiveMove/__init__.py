"""Top-level class for interacting with diveMove

The :class:`TDR` class aims to be a comprehensive class to encapsulate the
processing of `TDR` records from a data file.

This module instantiates an `R` session to interact with low-level
functions and methods of package `diveMove`.

Calibration and phase detection
-------------------------------

.. autosummary::

   TDR
   TDR.zoc
   TDR.detect_wet
   TDR.detect_dives
   TDR.detect_dive_phases
   TDR.calibrate_speed

Analyses
--------

.. autosummary::

   TDR.dive_stats
   TDR.time_budget
   TDR.stamp_dives

Plotting
--------

.. autosummary::

   TDR.plot
   TDR.plot_zoc
   TDR.plot_phases
   TDR.plot_dive_model

Functions
---------

.. autosummary::

   calibrate
   dump_config_template


Detect and quantify periods of major activities in a TDR record,
calibrating depth readings to generate summaries of diving behaviour. These
procedures include wet/dry phase detection, zero-offset correction (ZOC) of
depth, detection of dives, as well as proper labelling of the latter, among
others utilities for the analysis of TDR records.

All core procedures are encapsulated in :class:`TDR`, and are controlled by
a set of user-defined variables in a configuration file, and systematically
executed systematically by function :func:`calibrate`. This function can be
used as a template for custom processing using different
sequences. User-defined variables can also be directly specified as a
dictionary, which is a class attribute. The standard approach follows the
logical processing sequence described below:

1. Zero-offset correction
2. Detection of wet phases
3. Detection of dives
4. Detection of dive phases
5. Speed calibration (if required)
6. Calculation of statistics

Class :class:`TDR` methods implement these steps.

"""

from skdiveMove.tdr import TDR, calibrate
from skdiveMove.calibconfig import dump_config_template

__author__ = "Sebastian Luque <spluque@gmail.com>"
__license__ = "AGPLv3"
__version__ = "0.3.2.post1"
__all__ = ["TDR", "calibrate", "dump_config_template"]
