"""Representation of Inertial Measurement Unit data

"""

import numpy as np
import pandas as pd
import allantools as allan
import ahrs.filters as filters
from scipy import constants, signal, integrate
from sklearn import preprocessing
from skdiveMove.tdrsource import _load_dataset
from .allan import allan_coefs
from .vector import rotate_vector

_TIME_NAME = "timestamp"
_DEPTH_NAME = "depth"
_ACCEL_NAME = "acceleration"
_OMEGA_NAME = "angular_velocity"
_MAGNT_NAME = "magnetic_density"


class IMUBase:
    """Define IMU data source

    Use :class:`xarray.Dataset` to ensure pseudo-standard metadata.

    Attributes
    ----------
    imu_file : str
        String indicating the file where the data comes from.
    imu : xarray.Dataset
        Dataset with input data.
    imu_var_names : list
        Names of the data variables with accelerometer, angular velocity,
        and magnetic density measurements.
    has_depth : bool
        Whether input data include depth measurements.
    depth_name : str
        Name of the data variable with depth measurements.
    time_name : str
        Name of the time dimension in the dataset.
    quats : numpy.ndarray
        Array of quaternions representing the orientation relative to the
        frame of the IMU object data.  Note that the scalar component is
        last, following `scipy`'s convention.

    Examples
    --------
    This example illustrates some of the issues encountered while reading
    data files in a real-world scenario.  ``scikit-diveMove`` includes a
    NetCDF file with IMU signals collected using a Samsung Galaxy S5 mobile
    phone.  Set up instance from NetCDF example data:

    >>> import importlib.resources as rsrc
    >>> import xarray as xr
    >>> import skdiveMove.imutools as imutools
    >>> icdf = (rsrc.files("skdiveMove") / "tests" / "data" /
    ...         "samsung_galaxy_s5.nc")

    The angular velocity and magnetic density arrays have two sets of
    measurements: output and measured, which, along with the sensor axis
    designation, constitutes a multi-index.  These multi-indices can be
    rebuilt prior to instantiating IMUBase, as they provide significant
    advantages for indexing later:

    >>> s5ds = (xr.load_dataset(icdf)
    ...         .set_index(gyroscope=["gyroscope_type", "gyroscope_axis"],
    ...                    magnetometer=["magnetometer_type",
    ...                                  "magnetometer_axis"]))
    >>> imu = imutools.IMUBase(s5ds.sel(gyroscope="output",
    ...                                 magnetometer="output"),
    ...                        imu_filename=icdf)

    See :doc:`/demos/demo_allan` demo for an extended example of typical
    usage of the methods in this class.

    """
    def __init__(self, dataset,
                 acceleration_name=_ACCEL_NAME,
                 angular_velocity_name=_OMEGA_NAME,
                 magnetic_density_name=_MAGNT_NAME,
                 time_name=_TIME_NAME,
                 has_depth=False, depth_name=_DEPTH_NAME,
                 imu_filename=None):
        """Set up attributes for IMU objects

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset containing IMU sensor DataArrays, and optionally other
            DataArrays.
        acceleration_name : str, optional
            Name of the acceleration ``DataArray`` in the ``Dataset``.
        angular_velocity_name : str, optional
            Name of the angular velocity ``DataArray`` in the ``Dataset``.
        magnetic_density_name : str, optional
            Name of the magnetic density ``DataArray`` in the ``Dataset``.
        time_name : str, optional
            Name of the time dimension in the dataset.
        has_depth : bool, optional
            Whether input data include depth measurements.
        depth_name : str, optional
            Name of the depth ``DataArray`` in the ``Dataset``.
        imu_filename : str, optional
            Name of the file from which ``dataset`` originated.

        """
        self.time_name = time_name
        self.imu = dataset
        self.imu_var_names = [acceleration_name,
                              angular_velocity_name,
                              magnetic_density_name]
        if has_depth:
            self.has_depth = True
            self.depth_name = depth_name
        else:
            self.has_depth = False
            self.depth_name = None

        self.imu_file = imu_filename
        self.quats = None

    @classmethod
    def read_netcdf(cls, imu_file,
                    acceleration_name=_ACCEL_NAME,
                    angular_velocity_name=_OMEGA_NAME,
                    magnetic_density_name=_MAGNT_NAME,
                    time_name=_TIME_NAME,
                    has_depth=False, depth_name=_DEPTH_NAME,
                    **kwargs):
        """Instantiate object by loading Dataset from NetCDF file

        Provided all ``DataArray`` in the NetCDF file have the same
        dimensions (N, 3), this is an efficient way to instantiate.

        Parameters
        ----------
        imu_file : str
            As first argument for :func:`xarray.load_dataset`.
        acceleration_name : str, optional
            Name of the acceleration ``DataArray`` in the ``Dataset``.
        angular_velocity_name : str, optional
            Name of the angular velocity ``DataArray`` in the ``Dataset``.
        magnetic_density_name : str, optional
            Name of the magnetic density ``DataArray`` in the ``Dataset``.
        dimension_names : list, optional
            Names of the dimensions of the data in each of the sensors.
        has_depth : bool, optional
            Whether input data include depth measurements.
        depth_name : str, optional
            Name of the depth ``DataArray`` in the ``Dataset``.
        **kwargs
            Optional keyword arguments passed to
            :func:`xarray.load_dataset`.

        Returns
        -------
        obj : IMUBase
            Class matches the caller.

        """
        dataset = _load_dataset(imu_file, **kwargs)
        return cls(dataset, acceleration_name=acceleration_name,
                   angular_velocity_name=angular_velocity_name,
                   magnetic_density_name=magnetic_density_name,
                   time_name=time_name, has_depth=has_depth,
                   depth_name=depth_name, imu_filename=imu_file)

    def __str__(self):
        x = self.imu
        objcls = ("IMU -- Class {} object\n"
                  .format(self.__class__.__name__))
        src = "{0:<20} {1}\n".format("Source File", self.imu_file)
        imu_desc = "IMU: {}".format(x.__str__())

        return objcls + src + imu_desc

    def _allan_deviation(self, sensor, taus):
        """Compute Allan deviation for all axes of a given sensor

        Currently uses the modified Allan deviation in package
        `allantools`.

        Parameters
        ----------
        sensor : str
            Attribute name of the sensor of interest
        taus : float, str
            Tau value, in seconds, for which to compute statistic.  Can be
            one of "octave" or "decade" for automatic generation of the
            value.

        Returns
        -------
        pandas.DataFrame
            Allan deviation and error for each sensor axis.  DataFrame
            index is the averaging time `tau` for each estimate.

        """
        sensor_obj = getattr(self, sensor)
        sampling_rate = sensor_obj.attrs["sampling_rate"]
        sensor_std = preprocessing.scale(sensor_obj, with_std=False)

        allan_l = []
        for axis in sensor_std.T:
            taus, adevs, errs, ns = allan.mdev(axis, rate=sampling_rate,
                                               data_type="freq",
                                               taus=taus)
            # taus is common to all sensor axes
            adevs_df = pd.DataFrame(np.column_stack((adevs, errs)),
                                    columns=["allan_dev", "error"],
                                    index=taus)
            allan_l.append(adevs_df)

        keys = [sensor + "_" + i for i in list("xyz")]
        devs = pd.concat(allan_l, axis=1, keys=keys)
        return devs

    def allan_coefs(self, sensor, taus):
        """Estimate Allan deviation coefficients for each error type

        This procedure implements the autonomous regression method for
        Allan variance described in [1]_.

        Given averaging intervals ``taus`` and corresponding Allan
        deviation ``adevs``, compute the Allan deviation coefficient for
        each error type:

          - Quantization
          - (Angle, Velocity) Random Walk
          - Bias Instability
          - Rate Random Walk
          - Rate Ramp

        Parameters
        ----------
        sensor : str
            Attribute name of the sensor of interest
        taus : float, str
            Tau value, in seconds, for which to compute statistic.  Can be
            one of "octave" or "decade" for automatic generation of the
            value.

        Returns
        -------
        coefs_all : pandas.DataFrame
            Allan deviation coefficient and corresponding averaging time
            for each sensor axis and error type.
        adevs : pandas.DataFrame
            `MultiIndex` DataFrame with Allan deviation, corresponding
            averaging time, and fitted ARMAV model estimates of the
            coefficients for each sensor axis and error type.

        Notes
        -----
        Currently uses a modified Allan deviation formula.

        .. [1] Jurado, J, Schubert Kabban, CM, Raquet, J (2019).  A
               regression-based methodology to improve estimation of
               inertial sensor errors using Allan variance data. Navigation
               66:251-263.

        """
        adevs_errs = self._allan_deviation(sensor, taus)
        taus = adevs_errs.index.to_numpy()
        adevs = adevs_errs.xs("allan_dev", level=1, axis=1).to_numpy()

        coefs_l = []
        fitted_l = []
        for adevs_i in adevs.T:
            coefs_i, adevs_fitted = allan_coefs(taus, adevs_i)
            # Parse output for dataframe
            coefs_l.append(pd.Series(coefs_i))
            fitted_l.append(adevs_fitted)

        keys = [sensor + "_" + i for i in list("xyz")]
        coefs_all = pd.concat(coefs_l, keys=keys, axis=1)
        fitted_all = pd.DataFrame(np.column_stack(fitted_l), columns=keys,
                                  index=taus)
        fitted_all.columns = (pd.MultiIndex
                              .from_tuples([(c, "fitted")
                                            for c in fitted_all]))
        adevs = (pd.concat([adevs_errs, fitted_all], axis=1)
                 .sort_index(axis=1))
        return (coefs_all, adevs)

    def compute_orientation(self, method="Madgwick", **kwargs):
        """Compute the orientation of IMU tri-axial signals

        The method must be one of the following estimators implemented in
        Python module :mod:`ahrs.filters`:

          - ``AngularRate``: Attitude from angular rate
          - ``AQUA``: Algebraic quaternion algorithm
          - ``Complementary``: Complementary filter
          - ``Davenport``: Davenport's q-method
          - ``EKF``: Extended Kalman filter
          - ``FAAM``: Fast accelerometer-magnetometer combination
          - ``FLAE``: Fast linear attitude estimator
          - ``Fourati``: Fourati's nonlinear attitude estimation
          - ``FQA``: Factored quaternion algorithm
          - ``Madgwick``: Madgwick orientation filter
          - ``Mahony``: Mahony orientation filter
          - ``OLEQ``: Optimal linear estimator quaternion
          - ``QUEST``
          - ``ROLEQ``: Recursive optimal linear estimator of quaternion
          - ``SAAM``: Super-fast attitude from accelerometer and magnetometer
          - ``Tilt``: Attitude from gravity

        The estimated quaternions are stored in the ``quats`` attribute.

        Parameters
        ----------
        method : str, optional
            Name of the filtering method to use.
        **kwargs : optional keyword arguments
            Arguments passed to filtering method.

        """
        orienter_cls = getattr(filters, method)
        orienter = orienter_cls(acc=self.acceleration,
                                gyr=self.angular_velocity,
                                mag=self.magnetic_density,
                                Dt=self.sampling_interval,
                                **kwargs)

        self.quats = orienter.Q

    def dead_reckon(self, g=constants.g, Wn=1.0, k=1.0):
        """Calculate position assuming orientation is already known

        Integrate dynamic acceleration in the body frame to calculate
        velocity and position.  If the IMU instance has a depth signal, it
        is used in the integration instead of acceleration in the vertical
        dimension.

        Parameters
        ----------
        g : float, optional
            Assume gravity (:math:`m / s^2`) is equal to this value.
            Default to standard gravity.
        Wn : float, optional
            Cutoff frequency for second-order Butterworth lowpass filter.
        k : float, optional
            Scalar to apply to scale lowpass-filtered dynamic acceleration.
            This scaling has the effect of making position estimates
            realistic for dead-reckoning tracking purposes.

        Returns
        -------
        vel, pos : numpy.ndarray
            Velocity and position 2D arrays.

        """
        # Acceleration, velocity, and position from q and the measured
        # acceleration, get the \frac{d^2x}{dt^2}.  Retrieved sampling
        # frequency assumes common frequency
        fs = self.acceleration.attrs["sampling_rate"]
        # Shift quaternions to scalar last to match convention
        quats = np.roll(self.quats, -1, axis=1)
        g_v = rotate_vector(np.array([0, 0, g]), quats, inverse=True)
        acc_sensor = self.acceleration - g_v
        acc_space = rotate_vector(acc_sensor, quats, inverse=False)
        # Low-pass Butterworth filter design
        b, a = signal.butter(2, Wn, btype="lowpass", output="ba", fs=fs)
        acc_space_f = signal.filtfilt(b, a, acc_space, axis=0)
        # Position and Velocity through integration, assuming 0-velocity at t=0
        vel = integrate.cumulative_trapezoid(acc_space_f / k, dx=1.0 / fs,
                                             initial=0, axis=0)
        # Use depth derivative (on FLU) for the vertical dimension
        if self.has_depth:
            pos_z = self.depth
            zdiff = np.append([0], np.diff(pos_z))
            vel[:, -1] = -zdiff
            pos = np.nan * np.ones_like(acc_space)
            pos[:, -1] = pos_z
            pos[:, :2] = (integrate
                          .cumulative_trapezoid(vel[:, :2], dx=1.0 / fs,
                                                axis=0, initial=0))
        else:
            pos = integrate.cumulative_trapezoid(vel, dx=1.0 / fs,
                                                 axis=0, initial=0)

        return vel, pos

    def _get_acceleration(self):
        # Acceleration name is the first
        return self.imu[self.imu_var_names[0]]

    acceleration = property(_get_acceleration)
    """Return acceleration array

    Returns
    -------
    xarray.DataArray

    """

    def _get_angular_velocity(self):
        # Angular velocity name is the second
        return self.imu[self.imu_var_names[1]]

    angular_velocity = property(_get_angular_velocity)
    """Return angular velocity array

    Returns
    -------
    xarray.DataArray

    """

    def _get_magnetic_density(self):
        # Magnetic density name is the last one
        return self.imu[self.imu_var_names[-1]]

    magnetic_density = property(_get_magnetic_density)
    """Return magnetic_density array

    Returns
    -------
    xarray.DataArray

    """

    def _get_depth(self):
        return getattr(self.imu, self.depth_name)

    depth = property(_get_depth)
    """Return depth array

    Returns
    -------
    xarray.DataArray

    """

    def _get_sampling_interval(self):
        # Retrieve sampling rate from one DataArray
        sampling_rate = self.acceleration.attrs["sampling_rate"]
        sampling_rate_units = (self.acceleration
                               .attrs["sampling_rate_units"])

        if sampling_rate_units.lower() == "hz":
            itvl = 1.0 / sampling_rate
        else:
            itvl = sampling_rate

        return itvl

    sampling_interval = property(_get_sampling_interval)
    """Return sampling interval

    Assuming all `DataArray`s have the same interval, the sampling interval
    is retrieved from the acceleration `DataArray`.

    Returns
    -------
    xarray.DataArray

    Warnings
    --------
    The sampling rate is retrieved from the attribute named `sampling_rate`
    in the NetCDF file, which is assumed to be in Hz units.

    """
