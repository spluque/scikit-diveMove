#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up
import pkg_resources as pkg_rsrc
import matplotlib.pyplot as plt
import skdiveMove as skdive

# Declare figure sizes
_FIG1X1 = (11, 5)
_FIG2X1 = (10, 8)
_FIG3X1 = (11, 11)


# In[2]:


import numpy as np   # only for setting print options here
import pandas as pd  # only for setting print options here
import xarray as xr  # only for setting print options here

pd.set_option("display.precision", 3)
np.set_printoptions(precision=3, sign="+")
xr.set_options(display_style="html")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


ifile = (pkg_rsrc
         .resource_filename("skdiveMove",
                            ("tests/data/"
                             "ag_mk7_2002_022.nc")))
tdrX = skdive.TDR.read_netcdf(ifile, depth_name="depth", has_speed=True)
# Or simply use function ``skdive.tests.diveMove2skd`` to do the
# same with this particular data set.
print(tdrX)


# In[4]:


tdrX.get_depth("measured")


# In[5]:


tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
          depth_lim=[-1, 95], figsize=_FIG1X1);


# In[6]:


ccvars = ["light", "speed"]
tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
          depth_lim=[-1, 95], concur_vars=ccvars, figsize=_FIG3X1);


# In[7]:


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


# In[8]:


tdrX.detect_wet(dry_thr=pars["dry_thr"], wet_thr=pars["wet_thr"])


# In[9]:


tdrX.detect_dives(dive_thr=pars["dive_thr"])


# In[10]:


tdrX.detect_dive_phases(dive_model=pars["dive_model"],
                        smooth_par=pars["smooth_par"],
                        knot_factor=pars["knot_factor"],
                        descent_crit_q=pars["descent_crit_q"],
                        ascent_crit_q=pars["ascent_crit_q"])

print(tdrX)


# In[11]:


help(skdive.calibrate)


# In[12]:


tdrX.plot_phases(diveNo=list(range(250, 300)), surface=True, figsize=_FIG1X1);


# In[13]:


# Plot dive model for a dive
tdrX.plot_dive_model(diveNo=20, figsize=(10, 10));


# In[14]:


fig, ax = plt.subplots(figsize=(7, 6))
# Consider only changes in depth larger than 2 m
tdrX.calibrate_speed(z=2, ax=ax)
print(tdrX.speed_calib_fit.summary())


# In[15]:


print(tdrX.get_depth("zoc"))


# In[16]:


print(tdrX.get_speed("calibrated"))


# In[17]:


# Time series of the wet/dry phases
print(tdrX.wet_dry)


# In[18]:


print(tdrX.get_phases_params("wet_dry")["dry_thr"])


# In[19]:


print(tdrX.get_phases_params("wet_dry")["wet_thr"])


# In[20]:


print(tdrX.get_dives_details("row_ids"))


# In[21]:


print(tdrX.get_dives_details("spline_derivs"))


# In[22]:


print(tdrX.get_dives_details("crit_vals"))


# In[23]:


print(tdrX.time_budget(ignore_z=True, ignore_du=False))


# In[24]:


print(tdrX.time_budget(ignore_z=True, ignore_du=True))


# In[25]:


print(tdrX.dive_stats())


# In[26]:


print(tdrX.stamp_dives())

