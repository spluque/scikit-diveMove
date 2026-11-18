"""Test configuration classes.

"""

import pytest
import skdiveMove.calibconfig as calibconfig

from contextlib import nullcontext


@pytest.mark.parametrize("zoc_method", ["offset", "filter", "foo"])
def test_config_tdr_zoc(zoc_method):

    valid_methods = ["offset", "filter"]

    if zoc_method in valid_methods:
        cfg = calibconfig.ConfigTDRZOC(
            method=zoc_method)
        assert isinstance(cfg, calibconfig.ConfigTDRZOC)
    else:
        with pytest.raises(ValueError):
            calibconfig.ConfigTDRZOC(method=zoc_method)


@pytest.mark.parametrize(
    "dry_thr,wet_thr,expectation",
    [[1, 1, nullcontext(calibconfig.ConfigTDRWetDry(dry_thr=1, wet_thr=1))],
     [-1, 1, pytest.raises(ValueError)],
     [1, -1, pytest.raises(ValueError)]])
def test_config_tdr_wetdry(dry_thr, wet_thr, expectation):
    """Test combinations of valid and invalid inputs."""
    with expectation as e:
        assert calibconfig.ConfigTDRWetDry(
            dry_thr=dry_thr,
            wet_thr=wet_thr) == e


@pytest.mark.parametrize(
    "dive_thr,dive_model,knot_factor,smooth_par,"
    "descent_crit_q,ascent_crit_q,expectation",
    [[1, "unimodal", 1, 0.5, 0.5, 0.5,
      nullcontext(calibconfig.ConfigTDRDives(
          dive_thr=1, dive_model="unimodal", knot_factor=1, smooth_par=0.5,
          descent_crit_q=0.5, ascent_crit_q=0.5))],
     [1, "foo", 1, 0.5, 0.5, 0.5,
      pytest.raises(ValueError)],
     [1, "smooth.spline", 0, 0.5, 0.5, 0.5,
      pytest.raises(ValueError)],
     [1, "smooth.spline", 1, 0, 0.5, 0.5,
      pytest.raises(ValueError)],
     [1, "smooth.spline", 1, 0.5, 1.5, 0.5,
      pytest.raises(ValueError)],
     [1, "smooth.spline", 1, 0.5, 0.5, 1.5,
      pytest.raises(ValueError)]])
def test_config_tdr_dives(dive_thr, dive_model, knot_factor, smooth_par,
                          descent_crit_q, ascent_crit_q, expectation):
    """Test combinations of valid and invalid inputs."""
    with expectation as e:
        assert calibconfig.ConfigTDRDives(
            dive_thr=dive_thr,
            dive_model=dive_model, knot_factor=knot_factor,
            smooth_par=smooth_par, descent_crit_q=descent_crit_q,
            ascent_crit_q=ascent_crit_q) == e


@pytest.mark.parametrize(
    "tau,contour_level,expectation",
    [[0.5, 0.5, nullcontext(calibconfig.ConfigTDRSpeed(
        tau=0.5, contour_level=0.5))],
     [1.5, 0.5, pytest.raises(ValueError)],
     [0.5, 1.5, pytest.raises(ValueError)]])
def test_config_tdr_speed(tau, contour_level, expectation):
    """Test combinations of valid and invalid inputs."""
    with expectation as e:
        assert calibconfig.ConfigTDRSpeed(
            tau=tau, contour_level=contour_level) == e
