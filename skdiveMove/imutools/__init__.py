"""Tools and classes for common IMU data processing tasks

The :class:`IMUBase` represents tri-axial Inertial Measurement Unit (IMU)
data to study 3D kinematics.  IMU devices are often deployed together with
time-depth recorders (TDR).  The base class provides essential accessors
and other methods to perform error analysis and compute orientation using a
variety of algorithms, leveraging on the capabilities of Python package
AHRS.


Base class & methods summary
----------------------------

See `API` section for details on minor methods.

.. autosummary::

   IMUBase
   IMUBase.read_netcdf
   IMUBase.allan_coefs
   IMUBase.compute_orientation
   IMUBase.dead_reckon


Instrument frame to body frame transformations
----------------------------------------------

One of the first tasks in analysis of IMU data from devices mounted on wild
animals is estimating the orientation of the instrument on the animal.  The
:class:`IMU2Body` class facilitates this process for devices mounted on
air-breathing marine/aquatic animals that regularly come up to the surface
to breathe.  See Johnson (2011) for details on the approach.

Notes
~~~~~

A right-handed coordinate system is assumed in the input `IMU` data.

.. image:: .static/images/rhs_frame.png
   :scale: 40%

:class:`IMU2Body` subclass extends :class:`IMUBase`, providing an
integrated approach to estimating the relative orientation of two reference
frames: a) body (b) and b) sensor (s).  It adds the methods summarized
below.

.. image:: .static/images/imu2body_frames.png
   :scale: 40%

.. autosummary::

   IMU2Body
   IMU2Body.from_csv_nc
   IMU2Body.get_surface_vectors
   IMU2Body.get_orientation
   IMU2Body.get_orientations
   IMU2Body.orient_surfacing
   IMU2Body.orient_surfacings
   IMU2Body.orient_IMU
   IMU2Body.filter_surfacings
   IMU2Body.scatterIMU3D
   IMU2Body.tsplotIMU_depth


Calibration and error analysis
------------------------------

IMU measurements are generally affected by temperature, contain offsets and
are affected by the error characteristics of the sensors making the
measurements.  These need to be taken into account and the
:class:`IMUcalibrate` class provides a practical framework to do so.

.. autosummary::

   IMUcalibrate
   IMUcalibrate.build_tmodels
   IMUcalibrate.plot_experiment
   IMUcalibrate.plot_var_model
   IMUcalibrate.plot_standardized
   IMUcalibrate.get_model
   IMUcalibrate.get_offset
   IMUcalibrate.apply_model
   fit_ellipsoid
   apply_ellipsoid

API
---

"""

from .imu import IMUBase
from .imu2body import (IMU2Body, scatterIMU3D,
                       scatterIMU_svd, tsplotIMU_depth)
from .imucalibrate import IMUcalibrate
from .ellipsoid import fit_ellipsoid, apply_ellipsoid

__all__ = ["IMUBase", "IMU2Body", "IMUcalibrate",
           "fit_ellipsoid", "apply_ellipsoid",
           "scatterIMU_svd", "scatterIMU3D", "tsplotIMU_depth"]
