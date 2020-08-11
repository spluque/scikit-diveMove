"""Unit test for TDR classes

"""

import unittest as ut
import numpy as np
from pandas import DataFrame
from pandas.testing import assert_frame_equal
from skdiveMove.tdrphases import TDRPhases
from skdiveMove.tests import diveMove2skd


class TestTDRPhases(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        tdrX = diveMove2skd("TDRPhases")
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
        tdrX.zoc("offset", offset=zoc_offset)
        self.phases = tdrX

        tdrX = diveMove2skd("TDRPhases")
        tdrX.zoc("offset", offset=zoc_offset)
        tdrX.detect_wet(dry_thr=dry_thr, wet_cond=None,
                        wet_thr=wet_thr, interp_wet=False)
        tdrX.detect_dives(dive_thr)
        tdrX.detect_dive_phases(dive_model=dive_model,
                                smooth_par=smooth_par,
                                knot_factor=knot_factor,
                                descent_crit_q=descent_crit_q,
                                ascent_crit_q=ascent_crit_q)
        self.calib = tdrX
        diveNo_max = (tdrX
                      .get_dives_details("row_ids", "dive_id")
                      .max())
        self.diveNo_seq = np.arange(diveNo_max) + 1

    def test_init(self):
        self.assertIsInstance(self.phases, TDRPhases)

    def test_str(self):
        self.assertIn("Class TDRPhases object", self.phases.__str__())

    def test_detect_wet(self):
        # Bypass setUp's `calib` to do it from scratch
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        self.phases.detect_wet(dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr)
        wet_dry = self.phases.wet_dry
        self.assertIsInstance(wet_dry, DataFrame)
        self.assertEqual(wet_dry.ndim, 2)

        # Retest with interpolation of wet depth
        self.phases.detect_wet(dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr, interp_wet=True)
        wet_dry = self.phases.wet_dry
        self.assertIsInstance(wet_dry, DataFrame)
        self.assertEqual(wet_dry.ndim, 2)
        # we should have ZOC depth history updated
        depth_zoc = self.phases.depth_zoc
        self.assertIn("interp_wet", depth_zoc.attrs["history"])

        # Test providing wet_cond
        dry_cond = self.phases.depth.to_series().isna()
        self.phases.detect_wet(dry_thr=dry_thr, wet_cond=~dry_cond,
                               wet_thr=wet_thr, interp_wet=True)
        wwet_dry = self.phases.wet_dry
        # Here we expect same result as before
        assert_frame_equal(wet_dry, wwet_dry)

    def test_detect_dives(self):
        # Bypass setUp's `calib` to do it from scratch
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        self.phases.detect_wet(dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr, interp_wet=False)
        self.phases.detect_dives(dive_thr)
        row_ids = self.phases.get_dives_details("row_ids")
        self.assertIsInstance(row_ids, DataFrame)
        self.assertEqual(row_ids.ndim, 2)
        # self.assertEqual(row_ids.shape[0], self.tdrX.tdr.shape[0])

    def test_detect_dive_phases(self):
        # Bypass setUp's `calib` to do it from scratch
        dry_thr = self.default_pars["dry_thr"]
        wet_thr = self.default_pars["wet_thr"]
        dive_thr = self.default_pars["dive_thr"]
        dive_model = self.default_pars["dive_model"]
        smooth_par = self.default_pars["smooth_par"]
        knot_factor = self.default_pars["knot_factor"]
        descent_crit_q = self.default_pars["descent_crit_q"]
        ascent_crit_q = self.default_pars["ascent_crit_q"]
        self.phases.detect_wet(dry_thr=dry_thr, wet_cond=None,
                               wet_thr=wet_thr, interp_wet=False)
        self.phases.detect_dives(dive_thr)
        self.phases.detect_dive_phases(dive_model=dive_model,
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
        calib = self.calib
        budget = calib.time_budget(ignore_z=True, ignore_du=True)
        self.assertIsInstance(budget, DataFrame)
        budget = calib.time_budget(ignore_z=False, ignore_du=True)
        self.assertIsInstance(budget, DataFrame)
        budget = calib.time_budget(ignore_z=False, ignore_du=False)
        self.assertIsInstance(budget, DataFrame)
        budget = calib.time_budget(ignore_z=True, ignore_du=False)
        self.assertIsInstance(budget, DataFrame)

    def test_stamp_dives(self):
        calib = self.calib
        stamps = calib.stamp_dives(ignore_z=True)
        self.assertIsInstance(stamps, DataFrame)
        stamps = calib.stamp_dives(ignore_z=False)
        self.assertIsInstance(stamps, DataFrame)

    def test_get_dives_details(self):
        calib = self.calib
        # Test wrong key
        self.assertRaises(KeyError, calib.get_dives_details, "foo")
        self.assertRaises(KeyError, calib.get_dives_details,
                          "row_ids", "foo")

    def test_get_dive_deriv(self):
        calib = self.calib
        # random dive
        rdive = np.random.choice(self.diveNo_seq)

        # Full length derivative
        dder = calib.get_dive_deriv(rdive)
        self.assertIsInstance(dder, DataFrame)
        # Individual phases
        for phase in ["descent", "bottom", "ascent"]:
            dder = calib.get_dive_deriv(rdive, phase)
            self.assertIsInstance(dder, DataFrame)
            # Test nonexistent phase
            self.assertRaises(KeyError, calib.get_dive_deriv,
                              rdive, "foo")

    def test_get_params(self):
        calib = self.calib

        wet_dry = calib.get_phases_params("wet_dry")
        self.assertIsInstance(wet_dry, dict)
        dives = calib.get_phases_params("dives")
        self.assertIsInstance(dives, dict)
        self.assertRaises(KeyError, calib.get_phases_params, "foo")
        # ZOC params should be tuple
        self.assertIsInstance(calib.zoc_params, tuple)


if __name__ == '__main__':
    ut.main()
