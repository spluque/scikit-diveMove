"""Top-level class for interacting with diveMove

The :class:`TDR` class aims to be a comprehensive class to encapsulate the
processing of `TDR` records from a data file.

This module instantiates an `R` session to interact with low-level
functions and methods of package `diveMove`.

Class & Main Methods Summary
----------------------------

See `API` section for details on minor methods.

Calibration and phase detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   TDR
   TDR.zoc
   TDR.detect_wet
   TDR.detect_dives
   TDR.detect_dive_phases
   TDR.calibrate_speed

Analyses
~~~~~~~~

.. autosummary::

   TDR.dive_stats
   TDR.time_budget
   TDR.stamp_dives

Plotting
~~~~~~~~

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

API
---

"""

from skdiveMove.tdr import TDR, calibrate
from skdiveMove.calibconfig import dump_config_template

__author__ = "Sebastian Luque <spluque@gmail.com>"
__license__ = "AGPLv3"
__version__ = "0.2.2"
__all__ = ["TDR", "calibrate", "dump_config_template"]
