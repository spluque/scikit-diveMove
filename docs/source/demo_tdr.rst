.. _demo_tdr-label:

===========================
 Diving behaviour analysis
===========================

Here is a bird's-eye view of the functionality of `scikit-diveMove`,
loosely following `diveMove`'s `vignette`_.

.. _vignette: https://cran.r-project.org/web/packages/diveMove/vignettes/diveMove.pdf

Set up the environment.  Consider loading the `logging` module and setting
up a logger to monitor progress to this section.

.. jupyter-execute::

   # Set up
   import pkg_resources as pkg_rsrc
   import matplotlib.pyplot as plt
   import skdiveMove as skdive

   # Declare figure sizes
   _FIG1X1 = (11, 5)
   _FIG2X1 = (10, 8)
   _FIG3X1 = (11, 11)

.. jupyter-execute::
   :hide-code:
   :hide-output:

   import numpy as np   # only for setting print options here
   import pandas as pd  # only for setting print options here
   import xarray as xr  # only for setting print options here

   pd.set_option("display.precision", 3)
   np.set_printoptions(precision=3, sign="+")
   xr.set_options(display_style="html")
   %matplotlib inline


Reading data files
==================

Load `diveMove`'s example data, using ``TDR.__init__`` method, and print:

.. jupyter-execute::
   :linenos:

   ifile = (pkg_rsrc
            .resource_filename("skdiveMove",
                               ("tests/data/"
                                "ag_mk7_2002_022.nc")))
   tdrX = skdive.TDR.read_netcdf(ifile, depth_name="depth", has_speed=True)
   # Or simply use function ``skdive.tests.diveMove2skd`` to do the
   # same with this particular data set.
   print(tdrX)

Notice that `TDR` reads files in `NetCDF4`_ format, which is a very
versatile file format, and encourages using properly documented data sets.
`skdiveMove` relies on `xarray.Dataset` objects to represent such data
sets.  It is easy to generate a `xarray.Dataset` objects from Pandas
DataFrames by using method :meth:`.to_xarray`. `skdiveMove` documents
processing steps by appending to the `history` attribute, in an effort
towards building metadata standards.

.. _NetCDF4: https://www.unidata.ucar.edu/software/netcdf

Access measured data:

.. jupyter-execute::
   :linenos:

   tdrX.get_depth("measured")


Plot measured data:

.. jupyter-execute::
   :linenos:

   tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
             depth_lim=[-1, 95], figsize=_FIG1X1);

Plot concurrent data:

.. jupyter-execute::
   :linenos:

   ccvars = ["light", "speed"]
   tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
             depth_lim=[-1, 95], concur_vars=ccvars, figsize=_FIG3X1);


Calibrate measurements
======================

Calibration of TDR measurements involves the following steps, which rely on
data from pressure sensors (barometers):


Zero offset correction (ZOC) of depth measurements
--------------------------------------------------

Using the "offset" method here for speed performance reasons:

.. jupyter-execute::
   :linenos:

   # Helper dict to set parameter values
   pars = {"offset_zoc": 3,
           "dry_thr": 70,
           "wet_thr": 3610,
           "dive_thr": 3,
           "dive_model": "unimodal",
           "smooth_par": 0.1,
           "knot_factor": 3,
           "descent_crit_q": 0,
           "ascent_crit_q": 0}

   tdrX.zoc("offset", offset=pars["offset_zoc"])

   # Plot ZOC job
   tdrX.plot_zoc(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
                 figsize=(13, 6));


Detection of wet vs dry phases
------------------------------

Periods of missing depth measurements longer than `dry_thr` are considered
dry phases, whereas periods that are briefer than `wet_thr` are not
considered to represent a transition to a wet phase.

.. jupyter-execute::
   :linenos:

   tdrX.detect_wet(dry_thr=pars["dry_thr"], wet_thr=pars["wet_thr"])

Other options, not explored here, include providing a boolean mask Series
to indicate which periods to consider wet phases (argument `wet_cond`), and
whether to linearly interpolate depth through wet phases with duration
below `wet_thr` (argument `interp_wet`).


Detection of dive events
------------------------

When depth measurements are greater than `dive_thr`, a dive event is deemed
to have started, ending when measurements cross that threshold again.

.. jupyter-execute::
   :linenos:

   tdrX.detect_dives(dive_thr=pars["dive_thr"])


Detection of dive phases
------------------------

Two methods for dive phase detection are available ("unimodal" and
"smooth_spline"), and this demo uses the default "unimodal" method:

.. jupyter-execute::
   :linenos:

   tdrX.detect_dive_phases(dive_model=pars["dive_model"],
                           smooth_par=pars["smooth_par"],
                           knot_factor=pars["knot_factor"],
                           descent_crit_q=pars["descent_crit_q"],
                           ascent_crit_q=pars["ascent_crit_q"])

   print(tdrX)

Alternatively, all these steps can be performed together via the
`calibrate` function:

.. jupyter-execute::
   :linenos:

   help(skdive.calibrate)

which is demonstrated in :ref:`demo_bouts-label`.


Plot dive phases
----------------

Once TDR data are properly calibrated and phases detected, results can be
visualized:

.. jupyter-execute::
   :linenos:

   tdrX.plot_phases(diveNo=list(range(250, 300)), surface=True, figsize=_FIG1X1);

.. jupyter-execute::
   :linenos:

   # Plot dive model for a dive
   tdrX.plot_dive_model(diveNo=20, figsize=(10, 10));


Calibrate speed measurements
----------------------------

In addition to the calibration procedure described above, other variables
in the data set may also need to be calibrated.  `skdiveMove` provides
support for calibrating speed sensor data, by taking advantage of its
relationship with the rate of change in depth in the vertical dimension.

.. jupyter-execute::

   fig, ax = plt.subplots(figsize=(7, 6))
   # Consider only changes in depth larger than 2 m
   tdrX.calibrate_speed(z=2, ax=ax)
   print(tdrX.speed_calib_fit.summary())

Notice processing steps have been appended to the `history` attribute of
the `DataArray`:

.. jupyter-execute::

   print(tdrX.get_depth("zoc"))

.. jupyter-execute::

   print(tdrX.get_speed("calibrated"))


Access attributes of `TDR` instance
===================================

Following calibration, use the different accessor methods:

.. jupyter-execute::

   # Time series of the wet/dry phases
   print(tdrX.wet_dry)

.. jupyter-execute::

   print(tdrX.get_phases_params("wet_dry")["dry_thr"])

.. jupyter-execute::

   print(tdrX.get_phases_params("wet_dry")["wet_thr"])

.. jupyter-execute::

   print(tdrX.get_dives_details("row_ids"))

.. jupyter-execute::

   print(tdrX.get_dives_details("spline_derivs"))

.. jupyter-execute::

   print(tdrX.get_dives_details("crit_vals"))


Time budgets
============

.. jupyter-execute::

   print(tdrX.time_budget(ignore_z=True, ignore_du=False))

.. jupyter-execute::

   print(tdrX.time_budget(ignore_z=True, ignore_du=True))


Dive statistics
===============

.. jupyter-execute::

   print(tdrX.dive_stats())


Dive stamps
===========

.. jupyter-execute::

   print(tdrX.stamp_dives())

Feel free to download a copy of this demo
(:jupyter-download:script:`demo_tdr`).
