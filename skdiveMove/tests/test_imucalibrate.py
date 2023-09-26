"""Unit test of imu2animal module

"""

import importlib.resources as rsrc
import unittest as ut
import numpy as np
import numpy.testing as npt
import matplotlib as mpl
import matplotlib.pyplot as plt
import skdiveMove.imutools as skimu
import xarray as xr
from skdiveMove.imutools.imu import _ACCEL_NAME, _OMEGA_NAME, _MAGNT_NAME


_ICDF = (rsrc.files("skdiveMove") / "tests" / "data" /
         "cats_temperature_calib.nc")


class TestIMUcalibrate(ut.TestCase):
    """Test `IMUcalibrate` methods

    """

    @classmethod
    def setUpClass(cls):
        # An instance with good accelerometer filtering
        pers = [slice("2021-09-20T09:00:00", "2021-09-21T10:33:00"),
                slice("2021-09-21T10:40:00", "2021-09-22T11:55:00"),
                slice("2021-09-22T12:14:00", "2021-09-23T11:19:00")]
        imucal = (skimu.IMUcalibrate
                  .read_netcdf(_ICDF, periods=pers,
                               axis_order=list("zxy"),
                               time_name="timestamp_utc",
                               has_depth=True))
        cls.imucal = imucal

        # Set up a fully calibrated objected for efficiency
        imucalibrated = skimu.IMUcalibrate(imucal.imu, periods=pers,
                                           axis_order=imucal.axis_order,
                                           time_name=imucal.time_name,
                                           has_depth=True)
        fs = 1                  # sampling_rate Hz
        win_len = int(2 * 60 * fs) - 1
        imucalibrated.build_tmodels(_ACCEL_NAME, use_axis_order=True,
                                    win_len=win_len)
        imucalibrated.build_tmodels(_OMEGA_NAME, use_axis_order=False,
                                    win_len=win_len)
        imucalibrated.build_tmodels(_MAGNT_NAME, use_axis_order=True,
                                    win_len=win_len)
        imucalibrated.build_tmodels("depth", use_axis_order=False,
                                    win_len=win_len)

        cls.imucalibrated = imucalibrated

    def test_init(self):
        pers = self.imucal.periods
        imu_xr = self.imucal.imu
        axis_order = self.imucal.axis_order
        imucal = skimu.IMUcalibrate(imu_xr, periods=pers,
                                    axis_order=axis_order)
        self.assertIsInstance(imucal, skimu.IMUcalibrate)

    def test_str(self):
        imucal = self.imucal
        self.assertIn("Periods:", imucal.__str__())

    def test_tmodels(self):
        imucal = self.imucal
        fs = 1                  # sampling_rate Hz
        win_len = int(2 * 60 * fs) - 1
        acc_cal = imucal.build_tmodels(_ACCEL_NAME, use_axis_order=True,
                                       win_len=win_len)
        gyro_cal = imucal.build_tmodels(_OMEGA_NAME,
                                        use_axis_order=False,
                                        win_len=win_len)
        mag_cal = imucal.build_tmodels(_MAGNT_NAME,
                                       use_axis_order=True,
                                       win_len=win_len)
        depth_cal = imucal.build_tmodels("depth",
                                         use_axis_order=False,
                                         win_len=win_len)
        npt.assert_array_equal(np.array([len(acc_cal),
                                         len(gyro_cal),
                                         len(mag_cal),
                                         len(depth_cal)]),
                               np.array([3, 3, 3, 3]))

    def test_plot_var_model(self):
        imucal = self.imucalibrated
        # Test plotting models
        fig_acc, axs_acc = imucal.plot_var_model(_ACCEL_NAME,
                                                 use_axis_order=True)
        self.assertIsInstance(fig_acc, mpl.figure.Figure)
        self.assertEqual(axs_acc.size, 3)
        plt.close()
        fig_omega, axs_omega = imucal.plot_var_model(_OMEGA_NAME,
                                                     use_axis_order=False)
        self.assertIsInstance(fig_omega, mpl.figure.Figure)
        self.assertEqual(axs_omega.size, 9)
        plt.close()
        fig_mag, axs_mag = imucal.plot_var_model(_MAGNT_NAME,
                                                 use_axis_order=True)
        self.assertIsInstance(fig_mag, mpl.figure.Figure)
        self.assertEqual(axs_mag.size, 3)
        plt.close()
        fig_depth, axs_depth = imucal.plot_var_model("depth",
                                                     use_axis_order=False)
        self.assertIsInstance(fig_depth, mpl.figure.Figure)
        self.assertEqual(axs_depth.size, 3)
        plt.close()

    def test_plot_standardized(self):
        imucal = self.imucalibrated
        # Test plotting standardized variables
        fig_acc, axs_acc, axs_temp_acc = imucal.plot_standardized(
            _ACCEL_NAME, use_axis_order=True, ref_val=9.8)
        self.assertIsInstance(fig_acc, mpl.figure.Figure)
        self.assertEqual(axs_acc.size, 3)
        plt.close()
        fig_omega, axs_omega, axs_temp_omega = imucal.plot_standardized(
            _OMEGA_NAME, use_axis_order=False)
        self.assertIsInstance(fig_omega, mpl.figure.Figure)
        self.assertEqual(axs_omega.size, 9)
        plt.close()
        fig_mag, axs_mag, axs_temp_mag = imucal.plot_standardized(
            _MAGNT_NAME, use_axis_order=True)
        self.assertIsInstance(fig_mag, mpl.figure.Figure)
        self.assertEqual(axs_mag.size, 3)
        plt.close()
        fig_depth, axs_depth, axs_temp_depth = imucal.plot_standardized(
            "depth", use_axis_order=False)
        self.assertIsInstance(fig_depth, mpl.figure.Figure)
        self.assertEqual(axs_depth.size, 3)
        plt.close()

    def test_plot_experiment(self):
        imucal = self.imucal
        fig, axs, axs_temp = imucal.plot_experiment(0, _ACCEL_NAME)
        self.assertIsInstance(fig, mpl.figure.Figure)
        self.assertEqual(len(axs), 3)
        self.assertEqual(len(axs), len(axs_temp))

    def test_get_offset(self):
        imucal = self.imucalibrated
        offset_depth = imucal.get_offset("depth", period=0, T_alpha=8,
                                         ref_val=0)
        self.assertIsInstance(offset_depth, float)
        offset_acc = imucal.get_offset(_ACCEL_NAME, period=1, T_alpha=8,
                                       ref_val=9.8, axis="x")
        self.assertIsInstance(offset_acc, float)
        offset_omega = imucal.get_offset(_OMEGA_NAME, period=2, T_alpha=8,
                                         ref_val=0, axis="y")
        self.assertIsInstance(offset_omega, float)
        offset_mag = imucal.get_offset(_MAGNT_NAME, period=0, T_alpha=8,
                                       ref_val=60, axis="z")
        self.assertIsInstance(offset_mag, float)

    def test_apply_model(self):
        imucal = self.imucalibrated
        acc_cal = imucal.apply_model(_ACCEL_NAME, imucal.imu, T_alpha=10,
                                     ref_vals=[-9.8, 9.8, -9.8],
                                     use_axis_order=True)
        self.assertIsInstance(acc_cal, xr.DataArray)
        omega_cal = imucal.apply_model(_OMEGA_NAME, imucal.imu, T_alpha=10,
                                       ref_vals=[0, 0, 0],
                                       use_axis_order=False,
                                       model_idx=[0, 2, 2])
        self.assertIsInstance(omega_cal, xr.DataArray)
        mag_cal = imucal.apply_model(_MAGNT_NAME, imucal.imu, T_alpha=10,
                                     use_axis_order=False,
                                     model_idx=[1, 2, 0])
        self.assertIsInstance(mag_cal, xr.DataArray)
        # No T_alpha for depth, so use mean temperature across Dataset
        depth_cal = imucal.apply_model("depth", imucal.imu,
                                       model_idx=0)
        self.assertIsInstance(depth_cal, xr.DataArray)


if __name__ == '__main__':
    ut.main()
