"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import xarray as xr
from pandas import DataFrame
from pandas.testing import assert_frame_equal
import statsmodels
import skdiveMove as skdive
from skdiveMove.tests import diveMove2skd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class TestTDR(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd()
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
        self.tdr_calib = diveMove2skd()
        (self.tdr_calib
         .calibrate(zoc_method="offset",
                    offset=zoc_offset,
                    dry_thr=dry_thr,
                    wet_thr=wet_thr,
                    dive_thr=dive_thr,
                    dive_model=dive_model,
                    smooth_par=smooth_par,
                    knot_factor=knot_factor,
                    descent_crit_q=descent_crit_q,
                    ascent_crit_q=ascent_crit_q))
        diveNo_max = (self.tdr_calib
                      .get_dives_details("row_ids", "dive_id")
                      .max())
        self.diveNo_seq = np.arange(diveNo_max) + 1

    def test_init(self):
        self.assertIsInstance(self.tdrX, skdive.TDR)

    def test_str(self):
        self.assertIn("Class TDR object", self.tdrX.__str__())

    def test_zoc_offset(self):
        offset = self.default_pars["zoc_offset"]
        self.tdrX.zoc("offset", offset=offset)
        self.assertEqual(self.tdrX.zoc_depth.method, "offset")
        self.assertIsInstance(self.tdrX.zoc_depth.depth, xr.DataArray)
        self.assertIn("offset", self.tdrX.zoc_depth.params)

    def test_detect_wet(self):
        offset = self.default_pars["zoc_offset"]
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        self.tdrX.zoc("offset", offset=offset)
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                             wet_thr=wet_thr, interp_wet=False)
        wet_act_phases = self.tdrX.wet_dry
        self.assertIsInstance(wet_act_phases, DataFrame)
        self.assertEqual(wet_act_phases.ndim, 2)

        # Retest with interpolation of wet depth
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                             wet_thr=wet_thr, interp_wet=True)
        wet_act_phases = self.tdrX.wet_dry
        self.assertIsInstance(wet_act_phases, DataFrame)
        self.assertEqual(wet_act_phases.ndim, 2)
        # we should have ZOC depth history updated
        zoc_depth = self.tdrX.get_depth("zoc")
        self.assertIn("interp_wet", zoc_depth.attrs["history"])

        # Test providing wet_cond
        dry_cond = self.tdrX.get_depth("measured").to_series().isna()
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=~dry_cond,
                             wet_thr=wet_thr, interp_wet=True)
        wwet_act_phases = self.tdrX.wet_dry
        # Here we expect same result as before
        assert_frame_equal(wet_act_phases, wwet_act_phases)

    def test_detect_dives(self):
        offset = self.default_pars["zoc_offset"]
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        self.tdrX.zoc("offset", offset=offset)
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                             wet_thr=wet_thr, interp_wet=False)
        self.tdrX.detect_dives(dive_thr=dive_thr)
        row_ids = self.tdrX.get_dives_details("row_ids")
        self.assertIsInstance(row_ids, DataFrame)
        self.assertEqual(row_ids.ndim, 2)

    def test_detect_dive_phases(self):
        offset = self.default_pars["zoc_offset"]
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]
        self.tdrX.zoc("offset", offset=offset)
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                             wet_thr=wet_thr, interp_wet=False)
        self.tdrX.detect_dives(dive_thr=dive_thr)
        self.tdrX.detect_dive_phases(dive_model=dive_model,
                                     smooth_par=smooth_par,
                                     knot_factor=knot_factor,
                                     descent_crit_q=descent_crit_q,
                                     ascent_crit_q=ascent_crit_q)
        dive_model_tdrX = self.tdrX.get_dives_details("model")
        self.assertEqual(dive_model, dive_model_tdrX)
        crit_vals = self.tdrX.get_dives_details("crit_vals")
        self.assertIsInstance(crit_vals, DataFrame)
        self.assertEqual(crit_vals.ndim, 2)
        dids_per_row = (self.tdrX.phases
                        .get_dives_details("row_ids", "dive_id"))
        dids_uniq = dids_per_row[dids_per_row > 0].unique()
        self.assertEqual(crit_vals.shape[0], dids_uniq.size)

    def test_calibrate(self):
        crit_vals = self.tdr_calib.phases.get_dives_details("crit_vals")
        self.assertIsInstance(crit_vals, DataFrame)
        self.assertEqual(crit_vals.ndim, 2)
        dids_per_row = (self.tdr_calib.phases
                        .get_dives_details("row_ids", "dive_id"))
        dids_uniq = dids_per_row[dids_per_row > 0].unique()
        self.assertEqual(crit_vals.shape[0], dids_uniq.size)

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

    def test_get_dives_details(self):
        tdr_calib = self.tdr_calib

        # Test wrong key
        self.assertRaises(KeyError, tdr_calib.get_dives_details, "foo")
        self.assertRaises(KeyError, tdr_calib.get_dives_details,
                          "row_ids", "foo")

    def test_get_dive_deriv(self):
        tdr_calib = self.tdr_calib
        # random dive
        rdive = np.random.choice(self.diveNo_seq)

        # Full length derivative
        dder = tdr_calib.get_dive_deriv(rdive)
        self.assertIsInstance(dder, DataFrame)
        # Individual phases
        for phase in ["descent", "bottom", "ascent"]:
            dder = tdr_calib.get_dive_deriv(rdive, phase)
            self.assertIsInstance(dder, DataFrame)
            # Test nonexistent phase
            self.assertRaises(KeyError, tdr_calib.get_dive_deriv,
                              rdive, "foo")

    def test_get_params(self):
        tdr_calib = self.tdr_calib
        wet_dry = tdr_calib.get_phases_params("wet_dry")
        self.assertIsInstance(wet_dry, dict)
        dives = tdr_calib.get_phases_params("dives")
        self.assertIsInstance(dives, dict)
        self.assertRaises(KeyError, tdr_calib.get_phases_params, "foo")
        # ZOC params should be tuple
        self.assertIsInstance(tdr_calib.zoc_params, tuple)

    def test_time_budget(self):
        tdr_calib = self.tdr_calib

        budget = tdr_calib.time_budget(ignore_z=True, ignore_du=True)
        self.assertIsInstance(budget, DataFrame)
        budget = tdr_calib.time_budget(ignore_z=False, ignore_du=True)
        self.assertIsInstance(budget, DataFrame)
        budget = tdr_calib.time_budget(ignore_z=False, ignore_du=False)
        self.assertIsInstance(budget, DataFrame)
        budget = tdr_calib.time_budget(ignore_z=True, ignore_du=False)
        self.assertIsInstance(budget, DataFrame)

    def test_stamp_dives(self):
        tdr_calib = self.tdr_calib
        z = 2
        tdr_calib.calibrate_speed(z=z, plot=False)

        stamps = tdr_calib.stamp_dives(ignore_z=True)
        self.assertIsInstance(stamps, DataFrame)
        stamps = tdr_calib.stamp_dives(ignore_z=False)
        self.assertIsInstance(stamps, DataFrame)

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
                                tdr.zoc_depth.filters.iloc[:, 0])
        # Second filter should be in second Axes (3rd line)
        filter1 = axs[1].get_lines()[2].get_xydata()[:, 1]
        np.testing.assert_equal(filter1,
                                tdr.zoc_depth.filters.iloc[:, 1])
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
