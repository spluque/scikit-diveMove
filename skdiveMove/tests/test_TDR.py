"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import xarray as xr
from pandas import DataFrame
import statsmodels
from skdiveMove.tdr import TDR
from skdiveMove.tests import diveMove2skd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class TestTDR(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd("TDR")
        zoc_offset = 3
        dry_thr = 70
        wet_thr = 3610
        dive_thr = 3
        dive_model = "unimodal"
        smooth_par = 0.1
        knot_factor = 3
        descent_crit_q = 0
        ascent_crit_q = 0
        self.default_pars = {"zoc_offset": zoc_offset,
                             "dry_thr": dry_thr,
                             "wet_thr": wet_thr,
                             "dive_thr": dive_thr,
                             "dive_model": dive_model,
                             "smooth_par": smooth_par,
                             "knot_factor": knot_factor,
                             "descent_crit_q": descent_crit_q,
                             "ascent_crit_q": ascent_crit_q}
        tdr_calib = diveMove2skd("TDR")
        tdr_calib.zoc(method="offset", offset=zoc_offset)
        tdr_calib.detect_wet(dry_thr=dry_thr, wet_thr=wet_thr)
        tdr_calib.detect_dives(dive_thr=dive_thr)
        tdr_calib.detect_dive_phases(dive_model=dive_model,
                                     smooth_par=smooth_par,
                                     knot_factor=knot_factor,
                                     descent_crit_q=descent_crit_q,
                                     ascent_crit_q=ascent_crit_q)
        self.tdr_calib = tdr_calib
        diveNo_max = (self.tdr_calib
                      .get_dives_details("row_ids", "dive_id")
                      .max())
        self.diveNo_seq = np.arange(diveNo_max) + 1

    def test_init(self):
        self.assertIsInstance(self.tdrX, TDR)

    def test_str(self):
        self.assertIn("Class TDR object", self.tdrX.__str__())

    def test_calibrate_speed(self):
        z = 2
        tdr_calib = self.tdr_calib

        tdr_calib.calibrate_speed(z=z, plot=False)

        self.assertIsInstance(tdr_calib.speed_calib_fit,
                              (statsmodels.regression.linear_model
                               .RegressionResultsWrapper))

    def test_dive_stats(self):
        speed_calib_z = 2
        tdr_calib = self.tdr_calib
        tdr_calib.calibrate_speed(z=speed_calib_z)
        dstats = tdr_calib.dive_stats()
        self.assertIsInstance(dstats, DataFrame)

    def test_get_depth(self):
        tdr_calib = self.tdr_calib
        mdepth = tdr_calib.get_depth("measured")
        zocdepth = tdr_calib.get_depth("zoc")

        self.assertNotIn("history", mdepth.attrs)
        self.assertIn("history", zocdepth.attrs)
        # Wrong request
        self.assertRaises(LookupError, tdr_calib.get_depth, "foo")

    def test_get_speed(self):
        tdr_calib = self.tdr_calib
        z = 2
        tdr_calib.calibrate_speed(z=z, plot=False)
        mspeed = tdr_calib.get_speed("measured")
        calspeed = tdr_calib.get_speed("calibrated")
        self.assertNotIn("history", mspeed.attrs)
        self.assertIn("history", calspeed.attrs)
        # Wrong requests
        self.assertRaises(LookupError, tdr_calib.get_speed, "foo")
        self.assertRaises(LookupError, self.tdrX.get_speed, "calibrated")

    def test_extract_dives(self):
        tdr_calib = self.tdr_calib
        z = 2
        tdr_calib.calibrate_speed(z=z, plot=False)
        # random dives
        rdives = np.random.choice(self.diveNo_seq, 20).tolist()

        dives = tdr_calib.extract_dives(rdives, calib_depth=True,
                                        calib_speed=True)
        self.assertIsInstance(dives, xr.Dataset)
        dives = tdr_calib.extract_dives(rdives, calib_depth=True,
                                        calib_speed=False)
        self.assertIsInstance(dives, xr.Dataset)
        dives = tdr_calib.extract_dives(rdives, calib_depth=False,
                                        calib_speed=False)
        self.assertIsInstance(dives, xr.Dataset)
        dives = tdr_calib.extract_dives(rdives, calib_depth=False,
                                        calib_speed=True)
        self.assertIsInstance(dives, xr.Dataset)

    # Plotting tests

    def test_plot(self):
        tdr = self.tdrX
        fig, ax = tdr.plot()
        l_depth = ax.get_lines()[0].get_xydata()
        self.assertEqual(l_depth.shape[0],
                         tdr.get_depth("measured").shape[0])
        plt.close(fig)

        # With concur_vars
        fig, axs = tdr.plot(concur_vars="speed")
        l_speed = axs[1].get_lines()[0].get_xydata()
        self.assertEqual(l_speed.shape[0],
                         tdr.get_speed("measured").shape[0])
        plt.close(fig)

        fig, axs = tdr.plot(concur_vars="temperature",
                            concur_var_titles="Temperature")
        l_temp = axs[1].get_lines()[0].get_xydata()
        self.assertEqual(l_temp.shape[0],
                         tdr.get_speed("measured").shape[0])
        plt.close(fig)

    def test_plot_zoc(self):
        tdr_calib = self.tdr_calib
        fig, ax = tdr_calib.plot_zoc()
        lines = ax.get_lines()
        self.assertGreaterEqual(len(lines), 2)
        plt.close(fig)
        # TODO: test arguments

        # Calibrate with "filter" and test plot_zoc
        tdr = self.tdrX
        DB = [-2, 5]
        K = [3, 600]
        P = [0.5, 0.02]
        tdr.zoc(k=K, probs=P, depth_bounds=DB)
        fig, axs = tdr.plot_zoc(ylim=[-1, 10])
        # Measured depth should be in first Axes
        idepth = axs[0].get_lines()[1].get_xydata()[:, 1]
        np.testing.assert_equal(idepth,
                                tdr.get_depth("measured").values)
        # First filter should be in second Axes (2nd line)
        filter0 = axs[1].get_lines()[1].get_xydata()[:, 1]
        np.testing.assert_equal(filter0,
                                tdr.zoc_filters.iloc[:, 0])
        # Second filter should be in second Axes (3rd line)
        filter1 = axs[1].get_lines()[2].get_xydata()[:, 1]
        np.testing.assert_equal(filter1,
                                tdr.zoc_filters.iloc[:, 1])
        # ZOC depth should be in third Axes
        zdepth = axs[2].get_lines()[1].get_xydata()[:, 1]
        # Set < 0 as for ZOC
        zdepth[zdepth < 0] = 0
        np.testing.assert_equal(zdepth,
                                tdr.get_depth("zoc").values)

    def test_plot_phases(self):
        tdr_calib = self.tdr_calib
        fig, ax = tdr_calib.plot_phases()
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close(fig)
        # Test selected dives
        # random dives
        rdives = np.random.choice(self.diveNo_seq, 10).tolist()
        fig, ax = tdr_calib.plot_phases(diveNo=rdives)
        l_depth = ax.get_lines()[0].get_ydata()
        tdr_dives = (tdr_calib
                     .extract_dives(rdives, calib_depth=True,
                                    calib_speed=False))
        self.assertEqual(l_depth.size, tdr_dives.depth.size)
        plt.close(fig)

        # Test selected dives with surface
        fig, ax = tdr_calib.plot_phases(diveNo=rdives, surface=True)
        # Compare scatter collection against line data
        l_depth = ax.get_lines()[0].get_xydata()
        scat_data = ax.collections[0].get_offsets().data
        self.assertLessEqual(scat_data.size, l_depth.size)
        plt.close(fig)

        # Test selected dives with concur_vars
        fig, axs = tdr_calib.plot_phases(diveNo=rdives, concur_vars="speed")
        # We should have 2 Axes
        self.assertEqual(len(axs), 2)
        l_depth = axs[0].get_lines()[0].get_xydata()
        l_speed = axs[1].get_lines()[0].get_xydata()
        self.assertEqual(l_depth.size, l_speed.size)
        self.assertEqual(l_depth.shape[0], tdr_dives.depth.size)
        plt.close(fig)

        # Test selected dives with concur_vars and surface
        fig, axs = tdr_calib.plot_phases(diveNo=rdives, concur_vars="speed",
                                         surface=True)
        self.assertEqual(len(axs), 2)
        l_depth = axs[0].get_lines()[0].get_xydata()
        l_speed = axs[1].get_lines()[0].get_xydata()
        speed_scat = axs[1].collections[0].get_offsets().data
        self.assertEqual(l_depth.size, l_speed.size)
        self.assertLessEqual(speed_scat.size, l_speed.size)
        plt.close(fig)

    def test_plot_dive_model(self):
        tdr_calib = self.tdr_calib
        # Test selected random dives
        rdives = np.random.choice(self.diveNo_seq, 10).tolist()

        for dive in rdives:
            print("diveNo: {}".format(dive))
            fig, axs = tdr_calib.plot_dive_model(diveNo=dive)
            # TODO
            self.assertEqual(len(axs), 2)
            lines_ax0 = axs[0].get_lines()
            self.assertEqual(len(lines_ax0), 4)
            lines_ax1 = axs[1].get_lines()
            self.assertEqual(len(lines_ax1), 7)
            plt.close(fig)


if __name__ == '__main__':
    ut.main()
