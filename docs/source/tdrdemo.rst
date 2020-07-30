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
   import os
   import os.path as osp
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

   here = osp.dirname(os.getcwd())
   ifile = osp.join(here, "skdiveMove", "tests", "data", "ag_mk7_2002_022.nc")
   tdrX = skdive.TDR(ifile, depth_name="depth", has_speed=True)
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

   # Or simply use function ``skdive.tests.diveMove2skd`` to do the
   # same with this particular data set.


Plotting measured data
======================

.. jupyter-execute::
   :linenos:

   tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
             depth_lim=[95, -1], figsize=_FIG1X1);

Plot concurrent data:

.. jupyter-execute::
   :linenos:

   ccvars = ["light", "speed"]
   tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
             depth_lim=[95, -1], concur_vars=ccvars, figsize=_FIG3X1);


Calibrate measurements
======================

Depth measurements can be calibrated in a single step with the `.calibrate`
method:

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

   # Apply zero-offset correction with the "offset" method, and set other
   # parameters for detection of wet/dry phases and dive phases
   tdrX.calibrate(zoc_method="offset", offset=pars["offset_zoc"],
                  dry_thr=pars["dry_thr"],
                  wet_thr=pars["wet_thr"],
                  dive_thr=pars["dive_thr"],
                  dive_model=pars["dive_model"],
                  smooth_par=pars["smooth_par"],
                  knot_factor=pars["knot_factor"],
                  descent_crit_q=pars["descent_crit_q"],
                  ascent_crit_q=pars["ascent_crit_q"])

   # Plot ZOC job
   tdrX.plot_zoc(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
                 figsize=(13, 6));

Alternatively, each of the steps of the calibration process performed by
this method can be done in a stepwise manner, allowing finer control.
Please see the `TDR` class API section.


Plot dive phases
----------------

.. jupyter-execute::
   :linenos:

   tdrX.plot_phases(diveNo=list(range(250, 300)), surface=True, figsize=_FIG1X1);

.. jupyter-execute::
   :linenos:

   # Plot dive model for a dive
   tdrX.plot_dive_model(diveNo=20, figsize=(10, 10));


Access attributes of `TDR` instance
-----------------------------------

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


Calibrate speed measurements
----------------------------

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
(:jupyter-download:script:`tdrdemo`).
