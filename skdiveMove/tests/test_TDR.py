"""Unit test for TDR class

"""

import unittest as ut
# import numpy.testing as npt
import xarray as xr
from pandas import Series, DataFrame
import skdiveMove as skdive
from skdiveMove.tests import diveMove2skd


class TestTDR(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd()
        self.default_pars = {"offset_zoc": 3,
                             "dry_thr": 70,
                             "wet_thr": 3610,
                             "dive_thr": 3,
                             "dive_model": "unimodal",
                             "smooth_par": 0.1,
                             "knot_factor": 3,
                             "descent_crit_q": 0,
                             "ascent_crit_q": 0}

    def test_init(self):
        self.assertIsInstance(self.tdrX, skdive.TDR)

    def test_zoc_offset(self):
        self.tdrX.zoc("offset", offset=3)
        self.assertEqual(self.tdrX.zoc_depth.method, "offset")
        self.assertIsInstance(self.tdrX.zoc_depth.depth_zoc, xr.DataArray)
        self.assertIn("offset", self.tdrX.zoc_depth.params)

    def test_detect_wet(self):
        offset = self.default_pars["offset_zoc"]
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        self.tdrX.zoc("offset", offset=offset)
        self.tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                             wet_thr=wet_thr, interp_wet=False)
        wet_act_phases = self.tdrX.get_wet_activity()
        self.assertIsInstance(wet_act_phases, DataFrame)
        self.assertEqual(wet_act_phases.ndim, 2)
        # self.assertEqual(wet_act_phases.shape[0], self.tdrX.tdr.shape[0])
        # dry_thr_tdrX = self.tdrX.phases.get_wet_activity("dry_thr")
        # self.assertEqual(dry_thr_tdrX, dry_thr)
        # wet_thr_tdrX = self.tdrX.get_wet_activity("wet_thr")
        # self.assertEqual(wet_thr_tdrX, wet_thr)

    def test_detect_dives(self):
        offset = self.default_pars["offset_zoc"]
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
        # self.assertEqual(row_ids.shape[0], self.tdrX.tdr.shape[0])

    def test_detect_dive_phases(self):
        offset = self.default_pars["offset_zoc"]
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
        offset = self.default_pars["offset_zoc"]
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]

        self.tdrX.calibrate(zoc_method="offset", offset=offset,
                            dry_thr=dry_thr,
                            wet_thr=wet_thr,
                            dive_thr=dive_thr,
                            dive_model=dive_model,
                            smooth_par=smooth_par,
                            knot_factor=knot_factor,
                            descent_crit_q=descent_crit_q,
                            ascent_crit_q=ascent_crit_q)

        crit_vals = self.tdrX.phases.get_dives_details("crit_vals")
        self.assertIsInstance(crit_vals, DataFrame)
        self.assertEqual(crit_vals.ndim, 2)
        dids_per_row = (self.tdrX.phases
                        .get_dives_details("row_ids", "dive_id"))
        dids_uniq = dids_per_row[dids_per_row > 0].unique()
        self.assertEqual(crit_vals.shape[0], dids_uniq.size)


if __name__ == '__main__':
    ut.main()
