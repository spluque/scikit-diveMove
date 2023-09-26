"""Tests for utilities in the imutools subpackage

Tests for `imu2body` and `imucalibrate` modules are done separately.

"""

import warnings
import importlib.resources as rsrc
import unittest as ut
import numpy as np
import numpy.testing as npt
import xarray as xr
import skdiveMove.imutools as skimu
import skdiveMove.imutools.allan as skallan
from scipy.optimize import OptimizeWarning
from skdiveMove.imutools.ellipsoid import _ELLIPSOID_FTYPES


_ICDF0 = (rsrc.files("skdiveMove") / "tests" / "data" /
          "samsung_galaxy_s5.nc")
_ICDF1 = (rsrc.files("skdiveMove") / "tests" / "data" / "gertrude" /
          "magnt_accel_calib.nc")

_ICDF2 = (rsrc.files("skdiveMove") / "tests" / "data" / "gertrude" /
          "gert_imu_frame.nc")


class TestUtils(ut.TestCase):
    """Test `allan` functions

    """
    def setUp(self):
        cdf0 = (xr.load_dataset(_ICDF0)
                .set_index(gyroscope=["gyroscope_type", "gyroscope_axis"],
                           magnetometer=["magnetometer_type",
                                         "magnetometer_axis"]))
        imu = skimu.IMUBase(cdf0.sel(gyroscope="output",
                                     magnetometer="output"),
                            has_depth=False)
        self.imu = imu

        # Define taus
        maxn = np.floor(np.log2(imu.angular_velocity.shape[0] / 250))
        sampling_rate = imu.angular_velocity.attrs["sampling_rate"]
        omega_taus = ((50.0 / sampling_rate) *
                      np.logspace(0, int(maxn), 100, base=2.0))
        self.omega_taus = omega_taus

        # Ellipsoid data
        magnt_accel_xr = xr.load_dataset(_ICDF1)
        self.magnt_uncalib = magnt_accel_xr["magnetic_density"].to_numpy()
        self.accel_uncalib = magnt_accel_xr["acceleration"].to_numpy()

    def test_IMUBase(self):
        imu = skimu.IMUBase.read_netcdf(_ICDF2, has_depth=True)
        self.assertIsInstance(imu, skimu.IMUBase)
        self.assertIn("Class IMUBase object", imu.__str__())

    def test_get_devs(self):
        imu = self.imu
        omega_taus = self.omega_taus
        adevs = imu._allan_deviation("angular_velocity", taus=omega_taus)
        npt.assert_equal(adevs.shape[0], omega_taus.size)
        npt.assert_equal(adevs.shape[1], 6)

    def test_allan_coefs(self):
        imu = self.imu
        omega_taus = self.omega_taus
        adevs = imu._allan_deviation("angular_velocity", taus=omega_taus)

        # allan module function
        adev_x = adevs["angular_velocity_x"]["allan_dev"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            sigmas_d, adevs_reg = skallan.allan_coefs(omega_taus, adev_x)
        npt.assert_equal(len(sigmas_d), 5)
        npt.assert_equal(len(adevs_reg), len(omega_taus))

        # IMU method on sensor suite
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OptimizeWarning)
            allan_coefs, adevs_fit = imu.allan_coefs("angular_velocity",
                                                     taus=omega_taus)
        npt.assert_equal(allan_coefs.shape, (5, 3))
        npt.assert_equal(adevs_fit.shape, (len(omega_taus), 9))

    def test_ellipsoid(self):
        magnt = self.magnt_uncalib
        accel = self.accel_uncalib

        # Test the different types of ellipsoid fits with the magnetometers
        for ftype in _ELLIPSOID_FTYPES:
            off, gain, rotM = skimu.fit_ellipsoid(magnt, f=ftype)
            npt.assert_equal(off.size, gain.size)
            npt.assert_equal(off.size, 3)
            npt.assert_equal(rotM.shape, (3, 3))

        # Repeat fit and apply ellipsoid to accelerometers
        off, gain, rotM = skimu.fit_ellipsoid(accel, f="sxyz")
        accel_corr = skimu.apply_ellipsoid(accel, offset=off, gain=gain,
                                           rotM=rotM, ref_r=1.0)
        npt.assert_equal(accel.shape, accel_corr.shape)

    def test_compute_orientation(self):
        # The resulting orientation is the device, since this set up does
        # not convert to body frame
        imu = self.imu
        # Subset the dataset for speed
        imu.imu = imu.imu.sel(timestamp=slice(None, 500))
        # Default Madgwick, uses default AHRS gain parameter
        imu.compute_orientation()
        npt.assert_equal(imu.quats.shape,
                         [imu.acceleration.shape[0], 4])

    def test_dead_reckon(self):
        imu = self.imu
        # Subset the dataset for speed
        imu.imu = imu.imu.sel(timestamp=slice(None, 500))
        # Default Madgwick, uses default AHRS gain parameter
        imu.compute_orientation()
        vel, pos = imu.dead_reckon()
        npt.assert_equal(vel.shape, pos.shape)

        # Test with depth using _ICDF2, but resample for speed
        icdf = (xr.load_dataset(_ICDF2)
                .resample(timestamp="5s").interpolate("linear")
                .bfill("timestamp"))
        for var in icdf.data_vars:
            icdf[var].attrs["sampling_rate"] = 1 / 5
        imu = skimu.IMUBase(icdf, has_depth=True, imu_filename=_ICDF2)
        imu.compute_orientation()
        vel, pos = imu.dead_reckon(Wn=0.09)  # 0 < Wn < (fs / 2)
        npt.assert_equal(vel.shape, pos.shape)


if __name__ == '__main__':
    ut.main()
