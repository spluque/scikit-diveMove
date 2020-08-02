"""Unit test for TDR classes

"""

import unittest as ut
# import numpy.testing as npt
from pandas import DataFrame
import skdiveMove as skdive
from skdiveMove.tests import diveMove2skd


class TestTDRPhases(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        tdrX = diveMove2skd()
        self.default_pars = {"offset_zoc": 3,
                             "dry_thr": 70,
                             "wet_thr": 3610,
                             "dive_thr": 3,
                             "dive_model": "unimodal",
                             "smooth_par": 0.1,
                             "knot_factor": 3,
                             "descent_crit_q": 0,
                             "ascent_crit_q": 0}
        tdrX.zoc("offset", offset=self.default_pars["offset_zoc"])
        self.depth = tdrX.get_depth("zoc")
        self.phases = tdrX.phases

    def test_init(self):
        self.assertIsInstance(self.phases, skdive.tdrphases.TDRPhases)

    def test_detect_wet(self):
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        self.phases.detect_wet(self.depth, dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        wet_dry = self.phases.wet_dry
        self.assertIsInstance(wet_dry, DataFrame)
        self.assertEqual(wet_dry.ndim, 2)

    def test_detect_dives(self):
        depth = self.depth
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        self.phases.detect_wet(depth, dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        self.phases.detect_dives(depth, dive_thr)
        row_ids = self.phases.get_dives_details("row_ids")
        self.assertIsInstance(row_ids, DataFrame)
        self.assertEqual(row_ids.ndim, 2)
        # self.assertEqual(row_ids.shape[0], self.tdrX.tdr.shape[0])

    def test_detect_dive_phases(self):
        depth = self.depth
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]
        self.phases.detect_wet(depth, dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        self.phases.detect_dives(depth, dive_thr)
        self.phases.detect_dive_phases(depth, dive_model=dive_model,
                                       smooth_par=smooth_par,
                                       knot_factor=knot_factor,
                                       descent_crit_q=descent_crit_q,
                                       ascent_crit_q=ascent_crit_q)
        dive_model_tdrX = self.phases.get_dives_details("model")
        self.assertEqual(dive_model, dive_model_tdrX)
        crit_vals = self.phases.get_dives_details("crit_vals")
        self.assertIsInstance(crit_vals, DataFrame)
        self.assertEqual(crit_vals.ndim, 2)
        dids_per_row = (self.phases
                        .get_dives_details("row_ids", "dive_id"))
        dids_uniq = dids_per_row[dids_per_row > 0].unique()
        self.assertEqual(crit_vals.shape[0], dids_uniq.size)

    def test_time_budget(self):
        depth = self.depth
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]
        self.phases.detect_wet(depth, dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        self.phases.detect_dives(depth, dive_thr)
        self.phases.detect_dive_phases(depth, dive_model=dive_model,
                                       smooth_par=smooth_par,
                                       knot_factor=knot_factor,
                                       descent_crit_q=descent_crit_q,
                                       ascent_crit_q=ascent_crit_q)
        tbudget = self.phases.time_budget(ignore_z=True, ignore_du=True)
        self.assertIsInstance(tbudget, DataFrame)

    def test_stamp_dives(self):
        depth = self.depth
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]
        self.phases.detect_wet(depth, dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        self.phases.detect_dives(depth, dive_thr)
        self.phases.detect_dive_phases(depth, dive_model=dive_model,
                                       smooth_par=smooth_par,
                                       knot_factor=knot_factor,
                                       descent_crit_q=descent_crit_q,
                                       ascent_crit_q=ascent_crit_q)
        stamps = self.phases.stamp_dives(ignore_z=True)
        self.assertIsInstance(stamps, DataFrame)


if __name__ == '__main__':
    ut.main()
