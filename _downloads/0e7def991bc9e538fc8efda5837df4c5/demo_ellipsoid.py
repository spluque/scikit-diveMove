#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Set up
import pkg_resources as pkg_rsrc
import os.path as osp
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import skdiveMove.imutools as imutools
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# boiler plate stuff to help out
_FIG1X1 = (11, 5)
def gen_sphere(radius=1):
    """Generate coordinates on a sphere"""
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return (x, y, z)



np.set_printoptions(precision=3, sign="+")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


icdf = (pkg_rsrc
        .resource_filename("skdiveMove",
                           osp.join("tests", "data", "gertrude",
                                    "magnt_accel_calib.nc")))
magnt_accel = xr.load_dataset(icdf)
magnt = magnt_accel["magnetic_density"].to_numpy()
accel = magnt_accel["acceleration"].to_numpy()


# In[4]:


# Here, a symmetrical constraint whereby any plane passing through the
# origin is used, with all radii equal to each other
magnt_off, magnt_gain, _ = imutools.fit_ellipsoid(magnt, f="sxyz")
accel_off, accel_gain, _ = imutools.fit_ellipsoid(accel, f="sxyz")


# In[5]:


print("Magnetometer offsets [uT]: x={:.2f}, y={:.2f}, z={:.2f};"
      .format(*magnt_off),
      "gains [uT]: x={:.2f}, y={:.2f}, z={:.2f}".format(*magnt_gain))
print("Accelerometer offsets [g]: x={:.3f}, y={:.3f}, z={:.3f};"
      .format(*accel_off),
      "gains [g]: x={:.3f}, y={:.3f}, z={:.3f}".format(*accel_gain))


# In[6]:


magnt_refr = 56.9
magnt_corr = imutools.apply_ellipsoid(magnt, offset=magnt_off,
                                      gain=magnt_gain,
                                      rotM=np.diag(np.ones(3)),
                                      ref_r=magnt_refr)
accel_corr = imutools.apply_ellipsoid(accel, offset=accel_off,
                                      gain=accel_gain,
                                      rotM=np.diag(np.ones(3)),
                                      ref_r=1.0)


# In[7]:


magnt_refr_diff = [np.abs(magnt.max(axis=0)) - magnt_refr,
                   np.abs(magnt.min(axis=0)) - magnt_refr]
magnt_corr_refr_diff = [np.abs(magnt_corr.max(axis=0)) - magnt_refr,
                        np.abs(magnt_corr.min(axis=0)) - magnt_refr]

accel_refr_diff = [np.abs(accel.max(axis=0)) - 1.0,
                   np.abs(accel.min(axis=0)) - 1.0]
accel_corr_refr_diff = [np.abs(accel_corr.max(axis=0)) - 1.0,
                        np.abs(accel_corr.min(axis=0)) - 1.0]


# In[8]:


print("Uncorrected magnetometer difference to reference [uT]:")
print("maxima: x={:.2f}, y={:.2f}, z={:.2f};"
      .format(*magnt_refr_diff[0]),
      "minima: x={:.2f}, y={:.2f}, z={:.2f}"
      .format(*magnt_refr_diff[1]))
print("Corrected magnetometer difference to reference [uT]:")
print("maxima: x={:.2f}, y={:.2f}, z={:.2f};"
      .format(*magnt_corr_refr_diff[0]),
      "minima: x={:.2f}, y={:.2f}, z={:.2f}"
      .format(*magnt_corr_refr_diff[1]))

print("Uncorrected accelerometer difference to reference [g]:")
print("maxima: x={:.2f}, y={:.2f}, z={:.2f};"
      .format(*accel_refr_diff[0]),
      "minima: x={:.2f}, y={:.2f}, z={:.2f}"
      .format(*accel_refr_diff[1]))
print("Corrected accelerometer difference to reference [g]:")
print("maxima: x={:.2f}, y={:.2f}, z={:.2f};"
      .format(*accel_corr_refr_diff[0]),
      "minima: x={:.2f}, y={:.2f}, z={:.2f}"
      .format(*accel_corr_refr_diff[1]))


# In[9]:


_FIG1X2 = [13, 7]
fig = plt.figure(figsize=_FIG1X2)
ax0 = fig.add_subplot(121, projection="3d")
ax1 = fig.add_subplot(122, projection="3d")
ax0.set_xlabel(r"x [$\mu T$]")
ax0.set_ylabel(r"y [$\mu T$]")
ax0.set_zlabel(r"z [$\mu T$]")
ax1.set_xlabel(r"x [$g$]")
ax1.set_ylabel(r"y [$g$]")
ax1.set_zlabel(r"z [$g$]")

ax0.plot_surface(*gen_sphere(magnt_refr), rstride=4, cstride=4, color="c",
                 linewidth=0, alpha=0.3)
ax1.plot_surface(*gen_sphere(), rstride=4, cstride=4, color="c",
                 linewidth=0, alpha=0.3)
ax0.plot(magnt[:, 0], magnt[:, 1], magnt[:, 2],
         marker=".", linestyle="none", markersize=0.5,
         label="uncorrected")
ax0.plot(magnt_corr[:, 0], magnt_corr[:, 1], magnt_corr[:, 2],
         marker=".", linestyle="none", markersize=0.5,
         label="corrected")
ax1.plot(accel[:, 0], accel[:, 1], accel[:, 2],
         marker=".", linestyle="none", markersize=0.5,
         label="uncorrected")
ax1.plot(accel_corr[:, 0], accel_corr[:, 1], accel_corr[:, 2],
         marker=".", linestyle="none", markersize=0.5,
         label="corrected")
l1, lbl1 = fig.axes[-1].get_legend_handles_labels()
fig.legend(l1, lbl1, loc="lower center", borderaxespad=0, frameon=False,
           markerscale=12)
ax0.view_init(22, azim=-142)
ax1.view_init(22, azim=-142)
plt.tight_layout()
