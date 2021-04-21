#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up
import pkg_resources as pkg_rsrc
import pandas as pd
import matplotlib.pyplot as plt
from skdiveMove import calibrate
import skdiveMove.bouts as skbouts

# Declare figure sizes
_FIG1X1 = (7, 6)
_FIG1X2 = (12, 5)
_FIG3X1 = (11, 11)


# In[2]:


import numpy as np

pd.set_option("display.precision", 3)
np.set_printoptions(precision=3, sign="+")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


config_file = (pkg_rsrc
               .resource_filename("skdiveMove",
                                  ("config_examples/"
                                   "ag_mk7_2002_022_config.json")))
tdr_file = (pkg_rsrc
            .resource_filename("skdiveMove",
                               ("tests/data/"
                                "ag_mk7_2002_022.nc")))
tdrX = calibrate(tdr_file, config_file)
stats = tdrX.dive_stats()
stamps = tdrX.stamp_dives(ignore_z=True)
stats_tab = pd.concat((stamps, stats), axis=1)
stats_tab.info()


# In[4]:


postdives = stats_tab["postdive_dur"][stats_tab["phase_id"] == 4]
postdives_diff = postdives.dt.total_seconds().diff()[1:].abs()
# Remove isolated dives
postdives_diff = postdives_diff[postdives_diff < 2000]


# In[5]:


postdives_nlsbouts = skbouts.BoutsNLS(postdives_diff, 0.1)
print(postdives_nlsbouts)


# In[6]:


fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars2 = postdives_nlsbouts.init_pars([50], plot=True, ax=ax)


# In[7]:


coefs2, pcov2 = postdives_nlsbouts.fit(init_pars2)
# Coefficients
print(coefs2)


# In[8]:


# Covariance between parameters
print(pcov2)


# In[9]:


# `bec` returns ndarray, and we have only one here
print("bec = {[0]:.2f}".format(postdives_nlsbouts.bec(coefs2)))


# In[10]:


fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_nlsbouts.plot_fit(coefs2, ax=ax);


# In[11]:


fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars3 = postdives_nlsbouts.init_pars([50, 550], plot=True, ax=ax)


# In[12]:


coefs3, pcov3 = postdives_nlsbouts.fit(init_pars3)
# Coefficients
print(coefs3)


# In[13]:


# Covariance between parameters
print(pcov3)


# In[14]:


fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_nlsbouts.plot_fit(coefs3, ax=ax);


# In[15]:


fig, axs = plt.subplots(1, 2, figsize=_FIG1X2)
postdives_nlsbouts.plot_ecdf(coefs2, ax=axs[0])
postdives_nlsbouts.plot_ecdf(coefs3, ax=axs[1]);


# In[16]:


postdives_mlebouts = skbouts.BoutsMLE(postdives_diff, 0.1)
print(postdives_mlebouts)


# In[17]:


fig, ax = plt.subplots(figsize=_FIG1X1)
init_pars = postdives_mlebouts.init_pars([50], plot=True, ax=ax)


# In[18]:


p_bnd = (-2, None)                 # bounds for `p`
lda1_bnd = (-5, None)              # bounds for `lambda1`
lda2_bnd = (-10, None)             # bounds for `lambda2`
bnd1 = (p_bnd, lda1_bnd, lda2_bnd)
p_bnd = (1e-2, None)
lda1_bnd = (1e-4, None)
lda2_bnd = (1e-8, None)
bnd2 = (p_bnd, lda1_bnd, lda2_bnd)
fit1, fit2 = postdives_mlebouts.fit(init_pars,
                                    fit1_opts=dict(method="L-BFGS-B",
                                                   bounds=bnd1),
                                    fit2_opts=dict(method="L-BFGS-B",
                                                   bounds=bnd2))


# In[19]:


# First fit
print(fit1)


# In[20]:


# Second fit
print(fit2)


# In[21]:


print("bec = {:.2f}".format(postdives_mlebouts.bec(fit2)))


# In[22]:


fig, ax = plt.subplots(figsize=_FIG1X1)
postdives_mlebouts.plot_fit(fit2, ax=ax);


# In[23]:


fig, axs = plt.subplots(1, 2, figsize=_FIG1X2)
postdives_nlsbouts.plot_ecdf(coefs2, ax=axs[0])
axs[0].set_title("NLS")
postdives_mlebouts.plot_ecdf(fit2, ax=axs[1])
axs[1].set_title("MLE");


# In[24]:


bec = postdives_mlebouts.bec(fit2)
skbouts.label_bouts(postdives.dt.total_seconds(), bec, as_diff=True)

