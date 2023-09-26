"""Unit test of `imu2body` module

"""

import importlib.resources as rsrc
import os.path as osp
import tempfile
import unittest as ut
import numpy as np
import numpy.testing as npt
import skdiveMove.imutools as skimu
from skdiveMove.imutools import vector as skvector
from skdiveMove.imutools import imu2body as i2b
from skdiveMove.imutools.imu import _ACCEL_NAME, _OMEGA_NAME, _MAGNT_NAME


_ICDF = (rsrc.files("skdiveMove") / "tests" / "data" / "gertrude" /
         "gert_imu_frame.nc")
_ICSV = (rsrc.files("skdiveMove") / "tests" / "data" / "gertrude" /
         "gert_long_srfc.csv")


class TestIMU2Body(ut.TestCase):
    """Test `IMU2Body` methods

    """
    def setUp(self):
        # An instance with good accelerometer filtering
        self.imu2body = (skimu.IMU2Body
                         .from_csv_nc(_ICSV, imu_nc=_ICDF,
                                      endasc_col=0,
                                      beg_surface_col=5,
                                      end_surface_col=6,
                                      beg_next_desc_col=7,
                                      savgol_parms=(99, 2)))

    def test_str(self):
        imu2body = self.imu2body
        self.assertIn("Surface segment duration summary:",
                      imu2body.__str__())

    def test_get_surface_vectors(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[10]
        acc_idx = imu2body.get_surface_vectors(idx, "acceleration")
        # acc_idx_E = [-0.1770691, -0.2477986,  1.0155698]  # expected
        acc_idx_E = [-0.16186333, -0.2287527, 0.90136201]  # expected
        acc_idx_mu = acc_idx.mean(axis=0).to_numpy()
        npt.assert_almost_equal(acc_idx_mu, acc_idx_E)
        # Check smoothed acceleration
        acc_idx = imu2body.get_surface_vectors(idx, "acceleration",
                                               smoothed_accel=True)
        # acc_idx_E = [-0.1772635, -0.24425, 1.0174348]  # expected
        acc_idx_E = [-0.16205773, -0.2252041, 0.90322697]  # expected
        acc_idx_mu = acc_idx.mean(axis=0).to_numpy()
        npt.assert_almost_equal(acc_idx_mu, acc_idx_E)
        # Check getting magnetic density
        magnt_idx = imu2body.get_surface_vectors(idx, "magnetic_density")
        magnt_idx_E = [14.6451379, 5.4446945, -41.141773]  # expected
        magnt_idx_mu = magnt_idx.mean(axis=0).to_numpy()
        npt.assert_almost_equal(magnt_idx_mu, magnt_idx_E)
        # Check getting depth
        depth_idx = imu2body.get_surface_vectors(idx, "depth")
        depth_idx_E = 1.070827300498207  # expected
        depth_idx_mu = depth_idx.mean()
        npt.assert_almost_equal(depth_idx_mu, depth_idx_E)
        # Request non-sense
        self.assertRaises(ValueError, imu2body.get_surface_vectors,
                          idx, "foo")

    def test_get_orientation(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[20]
        Rctr2i, svd = imu2body.get_orientation(idx, plot=False,
                                               animate=False)
        # Rctr2i_E = np.array([[0.9843929, 0.104275, -0.1417652],
        #                      [-0.134066, 0.9661851, -0.2202559],
        #                      [0.1140042, 0.2358242, 0.9650855]])
        Rctr2i_E = np.array([[0.9842878, 0.10322875, -0.14325276],
                             [-0.13423075, 0.96454283, -0.22724268],
                             [0.11471545, 0.24290113, 0.96324421]])
        # svd_E = (np.array([[-0.9843929, -0.104275, -0.1417652],
        #                    [0.134066, -0.9661851, -0.2202559],
        #                    [-0.1140042, -0.2358242, 0.9650855]]),
        #          np.array([1.6949134e-02, 2.0848817e-03, 5.8511064e-05]),
        #          np.array([[-0.9843929, 0.134066, -0.1140042],
        #                    [-0.104275, -0.9661851, -0.2358242],
        #                    [-0.1417652, -0.2202559, 0.9650855]]))
        svd_E = (np.array([[-0.9842878, -0.10322875, -0.14325276],
                           [0.13423075, -0.96454283, -0.22724268],
                           [-0.11471545, -0.24290113, 0.96324421]]),
                 np.array([2.12607792e-02, 2.62058029e-03, 9.21715522e-05]),
                 np.array([[-0.9842878, 0.13423075, -0.11471545],
                           [-0.10322875, -0.96454283, -0.24290113],
                           [-0.14325276, -0.22724268, 0.96324421]]))
        npt.assert_almost_equal(Rctr2i.as_matrix(), Rctr2i_E)
        npt.assert_almost_equal(svd[0], svd_E[0])
        npt.assert_almost_equal(svd[1], svd_E[1])
        npt.assert_almost_equal(svd[2], svd_E[2])
        # Covariance of normalized (smoothed) acceleration in the
        # transformed frame for the selected surfacing segment should be
        # close to zero, as this is the one used to find the plane
        acci_sg = imu2body.get_surface_vectors(idx, _ACCEL_NAME,
                                               smoothed_accel=True)
        acci_sg_body = Rctr2i.apply(acci_sg, inverse=True)
        acci_sg_cov = np.cov(skvector.normalize(acci_sg_body),
                             rowvar=False)
        npt.assert_array_almost_equal(np.tril(acci_sg_cov, k=-1),
                                      np.zeros((3, 3)))

        # Plotting but not animating
        Rctr2i, svd = imu2body.get_orientation(idx, plot=True,
                                               animate=False)

    def test_get_orientations(self):
        imu2body = self.imu2body
        orientations = imu2body.get_orientations()
        euler_mu = orientations[["phi", "theta", "psi"]].mean(axis=0)
        # euler_mu_E = [-7.0749517, 8.0709853, -7.7198771]
        euler_mu_E = [-6.82113499, 8.10123712, -7.73541211]
        npt.assert_almost_equal(euler_mu, euler_mu_E)

    def test_orient_surfacing(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[33]
        Rctr2i, svd = imu2body.get_orientation(idx, plot=False,
                                               animate=False)
        imu_bodyi = imu2body.orient_surfacing(idx, Rctr2i)
        acci = imu_bodyi[_ACCEL_NAME]
        acci_mu = acci.mean(axis=0).to_numpy()
        # acci_mu_E = [-0.0243503, 0.0657713, 1.0543464]
        acci_mu_E = [-0.02463756, 0.06611085, 0.93899781]
        npt.assert_almost_equal(acci_mu, acci_mu_E)

    def test_orient_surfacings(self):
        imu2body = self.imu2body
        orients = imu2body.orient_surfacings()
        shape_E = [82800, 68]
        npt.assert_almost_equal(orients.sizes.get("timestamp"), shape_E[0])
        npt.assert_almost_equal(orients.sizes.get("endasc"), shape_E[1])

    def test_filter_surfacings(self):
        imu2body = self.imu2body
        imu2body.get_orientations()
        imu2body.filter_surfacings((0.04, 0.06))
        # srfc_shape_E = [23, 7]
        srfc_shape_E = [16, 3]
        npt.assert_equal(imu2body.surface_details.shape, srfc_shape_E)
        euler_mu = (imu2body.orientations[["phi", "theta", "psi"]]
                    .mean(axis=0))
        euler_mu_E = [-5.99329621, 7.1074814, -7.21119287]
        npt.assert_almost_equal(euler_mu, euler_mu_E)

    def test_orient_IMU(self):
        imu2body = self.imu2body
        imu2body.get_orientations()
        imu2body.filter_surfacings((0.04, 0.06))
        imus_body = imu2body.orient_IMU()
        shape_E = [171135, 16]
        npt.assert_almost_equal(imus_body.sizes.get("timestamp"),
                                shape_E[0])
        npt.assert_almost_equal(imus_body.sizes.get("endasc"),
                                shape_E[1])

    def test_scatterIMU3D(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[33]
        ax = imu2body.scatterIMU3D(idx, _ACCEL_NAME, animate=False)
        self.assertIsInstance(ax, i2b.plt.Axes)
        ax = imu2body.scatterIMU3D(idx, _ACCEL_NAME, animate=False,
                                   smoothed_accel=True)
        self.assertIsInstance(ax, i2b.plt.Axes)
        # Test plotting and animation
        with tempfile.TemporaryDirectory() as tmpdirname:
            anim_file = osp.join(tmpdirname, "gert_imu_{}.mp4".format(idx))
            ax = imu2body.scatterIMU3D(idx, _MAGNT_NAME,
                                       normalize=False, animate=True,
                                       animate_file=anim_file)
            self.assertIsInstance(ax, i2b.plt.Axes)
            assert osp.exists(anim_file)

    def test_tsplotIMU_depth(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[33]
        axs = imu2body.tsplotIMU_depth(_ACCEL_NAME, idx)
        for ax in axs:
            self.assertIsInstance(ax, i2b.plt.Axes)
        axs = imu2body.tsplotIMU_depth(_ACCEL_NAME, idx,
                                       smoothed_accel=True)
        for ax in axs:
            self.assertIsInstance(ax, i2b.plt.Axes)
        axs = imu2body.tsplotIMU_depth(_OMEGA_NAME)
        for ax in axs:
            self.assertIsInstance(ax, i2b.plt.Axes)

    def test_scatterIMU_svd(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[33]  # choose surface period
        acc_imu = imu2body.get_surface_vectors(idx, _ACCEL_NAME,
                                               smoothed_accel=True)
        Rctr2i, svd = imu2body.get_orientation(idx, plot=False,
                                               animate=False)

        with tempfile.TemporaryDirectory() as tmpdirname:
            anim_file = osp.join(tmpdirname, "gert_imu_{}.mp4".format(idx))
            ax = skimu.scatterIMU_svd(acc_imu, svd, Rctr2i, normalize=True,
                                      center=True, animate=True,
                                      animate_file=anim_file)
            self.assertIsInstance(ax, i2b.plt.Axes)
            assert osp.exists(anim_file)
            # Test the method
            Rctr2i, svd = imu2body.get_orientation(idx, plot=True,
                                                   animate=True,
                                                   animate_file=anim_file)


class TestTagTools(ut.TestCase):
    """Test `_TagTools` methods

    """
    def setUp(self):
        # An instance with good accelerometer filtering
        self.imu2body = (i2b._TagTools
                         .from_csv_nc(_ICSV, imu_nc=_ICDF,
                                      endasc_col=0,
                                      beg_surface_col=5,
                                      end_surface_col=6,
                                      beg_next_desc_col=7,
                                      savgol_parms=(99, 2)))

    def test_get_orientation(self):
        imu2body = self.imu2body
        idx = imu2body.surface_details.index[20]
        Rb2i, svd = imu2body.get_orientation(idx, plot=False,
                                             animate=False)
        Rb2i_E = np.array([[0.97028085, 0.18870411, -0.15147879],
                           [-0.21326930, 0.96263697, -0.16687200],
                           [0.11432966, 0.19421848, 0.97427302]])
        svd_E = (np.array([[-0.15131243, 0.97030681, 0.18870411],
                           [-0.16690856, -0.21324069, 0.96263697],
                           [0.97429260, 0.11416261, 0.19421848]]),
                 np.array([997.98763039, 26.70905699, 5.30331262]),
                 np.array([[-0.15131243, -0.16690856, 0.97429260],
                           [0.97030681, -0.21324069, 0.11416261],
                           [0.18870411, 0.96263697, 0.19421848]]))
        npt.assert_almost_equal(Rb2i.as_matrix(), Rb2i_E)
        npt.assert_almost_equal(svd[0], svd_E[0])
        npt.assert_almost_equal(svd[1], svd_E[1])
        npt.assert_almost_equal(svd[2], svd_E[2])

        # Plotting but not animating
        Rctr2i, svd = imu2body.get_orientation(idx, plot=True,
                                               animate=False)
        # Plotting with animation
        with tempfile.TemporaryDirectory() as tmpdirname:
            anim_file = osp.join(tmpdirname, "gert_imu_{}.mp4".format(idx))
            Rctr2i, svd = imu2body.get_orientation(idx, plot=True,
                                                   animate=True,
                                                   animate_file=anim_file)


if __name__ == '__main__':
    ut.main()
