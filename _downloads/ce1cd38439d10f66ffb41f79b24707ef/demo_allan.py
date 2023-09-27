#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up
import warnings
import importlib.resources as rsrc
import os.path as osp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import skdiveMove.imutools as imutools
from scipy.optimize import OptimizeWarning


# In[2]:


# boiler plate stuff to help out
import pandas as pd

_FIG1X1 = (11, 5)

np.set_printoptions(precision=3, sign="+")
pd.set_option("display.precision", 3)
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


icdf = (rsrc.files("skdiveMove") / "tests" /
        "data" / "samsung_galaxy_s5.nc")
s5ds = (xr.load_dataset(icdf)  # rebuild MultiIndex
        .set_index(gyroscope=["gyroscope_type", "gyroscope_axis"],
                   magnetometer=["magnetometer_type",
                                 "magnetometer_axis"]))
imu = imutools.IMUBase(s5ds.sel(gyroscope="measured",  # use multi-index
                                magnetometer="measured"),
                       imu_filename=icdf)


# In[4]:


fig, ax = plt.subplots(figsize=_FIG1X1)
lines = (imu.angular_velocity
         .plot.line(x="timestamp", add_legend=False, ax=ax))
ax.legend(lines, ["measured {}".format(i) for i in list("xyz")],
          loc=9, ncol=3, frameon=False, borderaxespad=0);


# In[5]:


maxn = np.floor(np.log2(imu.angular_velocity.shape[0] / 250))
taus = ((50.0 / imu.angular_velocity.attrs["sampling_rate"]) *
        np.logspace(0, int(maxn), 100, base=2.0))


# In[6]:


# Silence warning for inability to estimate parameter covariances, which
# is not a concern as we are not making inferences
with warnings.catch_warnings():
    warnings.simplefilter("ignore", OptimizeWarning)
    allan_coefs, adevs = imu.allan_coefs("angular_velocity", taus)

print(allan_coefs)


# In[7]:


import matplotlib.ticker as mticker

adevs_ad = adevs.xs("allan_dev", level=1, axis=1)
adevs_fit = adevs.xs("fitted", level=1, axis=1)
fig, ax = plt.subplots(figsize=[6, 5])
for sensor, coefs in adevs_ad.items():
    suffix = sensor.split("_")[-1]
    ax.loglog(adevs_ad.index, adevs_ad[sensor], marker=".",
              linestyle="none",
              label="measured {}".format(suffix))
for sensor, fitted in adevs_ad.items():
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

