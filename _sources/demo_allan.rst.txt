==========================
 Allan deviation analysis
==========================

Allan deviation analysis is a technique for quantifying the different
sources of error affecting IMU measurements.  The `IMU` class features
methods to fit a model to averaging times :math:`\tau` and Allan deviation
estimates :math:`\sigma`.

.. jupyter-execute::

   # Set up
   import warnings
   import pkg_resources as pkg_rsrc
   import os.path as osp
   import numpy as np
   import xarray as xr
   import matplotlib.pyplot as plt
   import skdiveMove.imutools as imutools
   from scipy.optimize import OptimizeWarning

.. jupyter-execute::
   :hide-code:

   # boiler plate stuff to help out
   import pandas as pd

   _FIG1X1 = (11, 5)

   np.set_printoptions(precision=3, sign="+")
   pd.set_option("display.precision", 3)
   %matplotlib inline

For demonstrating the methods available in the ``IMUBase`` class, IMU
measurements from an Android mobile phone were collected for 6 hours at 100
Hz frequency, but subsequently decimated to 10 Hz with a forward/backward
filter to avoid phase shift.  The phone was kept immobile on a table,
facing up, for the data collection period.  Note that two sets of
measurements for the magnetometer and gyroscope were recorded: output and
measured.  The type of measurement and the sensor axis constitute a
multi-index, which provide significant advantages for indexing, so these
are rebuilt:

.. jupyter-execute::
   :linenos:

   icdf = (pkg_rsrc
           .resource_filename("skdiveMove",
	                      osp.join("tests", "data",
			               "samsung_galaxy_s5.nc")))
   s5ds = (xr.load_dataset(icdf)  # rebuild MultiIndex
           .set_index(gyroscope=["gyroscope_type", "gyroscope_axis"],
                      magnetometer=["magnetometer_type",
                                    "magnetometer_axis"]))
   imu = imutools.IMUBase(s5ds.sel(gyroscope="measured",  # use multi-index
                                   magnetometer="measured"),
                          imu_filename=icdf)

Note that the data collected are uncorrected for bias.  It is unclear
whether Android's raw (measured) sensor data have any other corrections or
calibrations, but the assumption here is that none were performed.

.. jupyter-execute::
   :hide-code:

   fig, ax = plt.subplots(figsize=_FIG1X1)
   lines = (imu.angular_velocity
            .plot.line(x="timestamp", add_legend=False, ax=ax))
   ax.legend(lines, ["measured {}".format(i) for i in list("xyz")],
             loc=9, ncol=3, frameon=False, borderaxespad=0);

The first step of the analysis involves the calculation of :math:`\tau` and
:math:`\sigma`.  The choice of :math:`\tau` sequence is crucial.

.. jupyter-execute::
   :linenos:

   maxn = np.floor(np.log2(imu.angular_velocity.shape[0] / 250))
   taus = ((50.0 / imu.angular_velocity.attrs["sampling_rate"]) *
           np.logspace(0, int(maxn), 100, base=2.0))

These can be used for fitting the ARMAV model and retrieve the coefficients
for the five most common sources of variability (quantization,
angle/velocity random walk, bias instability, rate random walk, and rate
ramp).

.. jupyter-execute::
   :linenos:

   # Silence warning for inability to estimate parameter covariances, which
   # is not a concern as we are not making inferences
   with warnings.catch_warnings():
       warnings.simplefilter("ignore", OptimizeWarning)
       allan_coefs, adevs = imu.allan_coefs("angular_velocity", taus)

   print(allan_coefs)

.. jupyter-execute::
   :hide-code:

   import matplotlib.ticker as mticker

   adevs_ad = adevs.xs("allan_dev", level=1, axis=1)
   adevs_fit = adevs.xs("fitted", level=1, axis=1)
   fig, ax = plt.subplots(figsize=[6, 5])
   for sensor, coefs in adevs_ad.iteritems():
       suffix = sensor.split("_")[-1]
       ax.loglog(adevs_ad.index, adevs_ad[sensor], marker=".",
                 linestyle="none",
                 label="measured {}".format(suffix))
   for sensor, fitted in adevs_ad.iteritems():
       suffix = sensor.split("_")[-1]
       ax.loglog(adevs_fit.index, adevs_fit[sensor],
                 color="r", linewidth=4, alpha=0.4,
                 label="fitted {}".format(suffix))
   ax.yaxis.set_minor_formatter(mticker.LogFormatter())
   ax.set_title("Angular velocity Allan Deviation")
   ax.set_ylabel(r"$\sigma\ (^\circ/s$)")
   ax.set_xlabel(r"$\tau$ (s)")
   ax.grid(which="both")
   ax.legend(loc=9, frameon=False, borderaxespad=0, ncol=2);


Feel free to download a copy of this demo
(:jupyter-download:script:`demo_allan`).
