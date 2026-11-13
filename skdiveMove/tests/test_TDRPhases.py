"""Test TDRPhases class

"""

import numpy as np
import pytest

from pandas import DataFrame, Series
from pandas.testing import assert_frame_equal
from skdiveMove.tdrphases import TDRPhases
from skdiveMove.tests.scenarios import TDRScenario  # noqa: E402


@pytest.fixture
def test_tdr():
    return TDRScenario("TDRPhases", has_speed=True)


@pytest.fixture(scope="module")
def test_tdr_calib():
    tdr = TDRScenario("TDRPhases", has_speed=True)
    tdr.detect_dive_phases()
    return tdr


def test_init(test_tdr):
    assert isinstance(test_tdr.tdrX, TDRPhases)


def test_str(test_tdr):
    assert "Class TDRPhases object" in test_tdr.tdrX.__str__()


def test_detect_wet(test_tdr):
    test_tdr.detect_wet()
    wet_dry = test_tdr.tdrX.wet_dry
    assert isinstance(wet_dry, DataFrame)
    assert wet_dry.ndim == 2


def test_detect_wet_interp(test_tdr):
    default_pars = test_tdr.default_pars
    test_tdr.zoc()
    test_tdr.tdrX.detect_wet(
        dry_thr=default_pars["dry_thr"],
        wet_cond=None,
        wet_thr=default_pars["wet_thr"],
        interp_wet=True)
    wet_dry = test_tdr.tdrX.wet_dry
    assert isinstance(wet_dry, DataFrame)
    assert wet_dry.ndim == 2
    # we should have ZOC depth history updated
    depth_zoc = test_tdr.tdrX.depth_zoc
    assert "interp_wet" in depth_zoc.attrs["history"]

    # Test providing wet_cond
    dry_cond = test_tdr.tdrX.depth.to_series().isna()
    test_tdr.tdrX.detect_wet(
        dry_thr=default_pars["dry_thr"],
        wet_cond=~dry_cond,
        wet_thr=default_pars["wet_thr"],
        interp_wet=True)
    wwet_dry = test_tdr.tdrX.wet_dry
    # Here we expect same result as before
    assert_frame_equal(wet_dry, wwet_dry)


def test_detect_dives(test_tdr):
    test_tdr.detect_dives()
    row_ids = test_tdr.tdrX.get_dives_details("row_ids")
    assert isinstance(row_ids, DataFrame)
    assert row_ids.ndim == 2
    # self.assertEqual(row_ids.shape[0], self.tdrX.tdr.shape[0])


def test_detect_dive_phases(test_tdr_calib):
    dive_model_tdrX = test_tdr_calib.tdrX.get_dives_details("model")
    dive_model = test_tdr_calib.default_pars["dive_model"]
    assert dive_model == dive_model_tdrX
    crit_vals = test_tdr_calib.tdrX.get_dives_details("crit_vals")
    assert isinstance(crit_vals, DataFrame)
    assert crit_vals.ndim == 2
    dids_per_row = (test_tdr_calib.tdrX
                    .get_dives_details("row_ids", "dive_id"))
    dids_uniq = dids_per_row[dids_per_row > 0].unique()
    assert crit_vals.shape[0] == dids_uniq.size


def test_time_budget(test_tdr_calib):
    calib = test_tdr_calib.tdrX
    budget = calib.time_budget(ignore_z=True, ignore_du=True)
    assert isinstance(budget, DataFrame)
    budget = calib.time_budget(ignore_z=False, ignore_du=True)
    assert isinstance(budget, DataFrame)
    budget = calib.time_budget(ignore_z=False, ignore_du=False)
    assert isinstance(budget, DataFrame)
    budget = calib.time_budget(ignore_z=True, ignore_du=False)
    assert isinstance(budget, DataFrame)


def test_stamp_dives(test_tdr_calib):
    calib = test_tdr_calib.tdrX
    stamps = calib.stamp_dives(ignore_z=True)
    assert isinstance(stamps, DataFrame)
    stamps = calib.stamp_dives(ignore_z=False)
    assert isinstance(stamps, DataFrame)


def test_get_dives_details(test_tdr_calib):
    calib = test_tdr_calib.tdrX
    # Test wrong key
    with pytest.raises(KeyError):
        calib.get_dives_details("foo")
    with pytest.raises(KeyError):
        calib.get_dives_details("row_ids", "foo")


def test_get_dive_deriv(test_tdr_calib):
    calib = test_tdr_calib.tdrX
    # random dive
    rdive = np.random.choice(test_tdr_calib.diveNo_seq)

    # Full length derivative
    dder = calib.get_dive_deriv(rdive)
    assert isinstance(dder, DataFrame)
    # Individual phases
    for phase in ["descent", "bottom", "ascent"]:
        dder = calib.get_dive_deriv(rdive, phase)
        assert isinstance(dder, DataFrame)
        # Test nonexistent phase
        with pytest.raises(KeyError):
            calib.get_dive_deriv(rdive, "foo")


def test_get_params(test_tdr_calib):
    calib = test_tdr_calib.tdrX

    wet_dry = calib.get_phases_params("wet_dry")
    assert isinstance(wet_dry, dict)
    dives = calib.get_phases_params("dives")
    assert isinstance(dives, dict)
    with pytest.raises(KeyError):
        calib.get_phases_params("foo")
    # ZOC params should be tuple
    assert isinstance(calib.zoc_params, tuple)

    # dive spline slots
    assert isinstance(calib.
                      _get_dive_spline_slot(diveNo=300,
                                            name="data"),
                      Series)
    assert isinstance(calib.
                      _get_dive_spline_slot(diveNo=300,
                                            name="xy"),
                      Series)
    scalars = ["order", "lambda.opt", "sigmasq", "degree",
               "g", "a", "b", "variter"]
    for name in scalars:
        assert isinstance(calib.
                          _get_dive_spline_slot(diveNo=300,
                                                name=name),
                          float)
