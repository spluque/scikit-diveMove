"""Test TDR class

"""

import numpy as np
import xarray as xr
import statsmodels
import matplotlib as mpl
mpl.use("Agg")
import pytest                    # noqa: E402

from pandas import DataFrame    # noqa: E402
from skdiveMove.tdr import TDR  # noqa: E402
from skdiveMove.tests.scenarios import TDRScenario  # noqa: E402


@pytest.fixture(scope="module")
def test_tdr():
    return TDRScenario("TDR", has_speed=True)


@pytest.fixture(scope="module")
def test_tdr_calib():
    tdr = TDRScenario("TDR", has_speed=True)
    tdr.detect_dive_phases()
    return tdr


def test_init(test_tdr):
    assert isinstance(test_tdr.tdrX, TDR)


def test_str(test_tdr):
    assert "Class TDR object" in test_tdr.tdrX.__str__()


def test_calibrate_speed(test_tdr_calib):
    z = 2
    tdr_calib = test_tdr_calib.tdrX

    tdr_calib.calibrate_speed(z=z, plot=False)

    assert isinstance(tdr_calib.speed_calib_fit,
                      (statsmodels.regression.linear_model
                       .RegressionResultsWrapper))
    # Check whether coefficients are in __str__ method of original object
    assert "Speed calibration coefficients: " in tdr_calib.__str__()


def test_dive_stats(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    dstats = tdr_calib.dive_stats()
    assert isinstance(dstats, DataFrame)


def test_get_depth(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    mdepth = tdr_calib.get_depth("measured")
    zocdepth = tdr_calib.get_depth("zoc")

    assert "history" not in mdepth.attrs
    assert "history" in zocdepth.attrs
    # Wrong request
    with pytest.raises(LookupError):
        tdr_calib.get_depth("foo")


def test_get_speed(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    mspeed = tdr_calib.get_speed("measured")
    calspeed = tdr_calib.get_speed("calibrated")
    assert "history" not in mspeed.attrs
    assert "history" in calspeed.attrs
    # Wrong requests
    with pytest.raises(LookupError):
        tdr_calib.get_speed("foo")


def test_extract_dives(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    # random dives
    rdives = np.random.choice(test_tdr_calib.diveNo_seq, 20).tolist()

    dives = tdr_calib.extract_dives(rdives, calib_depth=True,
                                    calib_speed=True)
    assert isinstance(dives, xr.Dataset)
    dives = tdr_calib.extract_dives(rdives, calib_depth=True,
                                    calib_speed=False)
    assert isinstance(dives, xr.Dataset)
    dives = tdr_calib.extract_dives(rdives, calib_depth=False,
                                    calib_speed=False)
    assert isinstance(dives, xr.Dataset)
    dives = tdr_calib.extract_dives(rdives, calib_depth=False,
                                    calib_speed=True)
    assert isinstance(dives, xr.Dataset)


# Plotting tests

@pytest.mark.plotting
def test_plot(test_tdr):
    tdr = test_tdr.tdrX
    fig, ax = tdr.plot()
    l_depth = ax.get_lines()[0].get_xydata()
    assert l_depth.shape[0] == tdr.get_depth("measured").shape[0]

    # With concur_vars
    fig, axs = tdr.plot(concur_vars="speed")
    l_speed = axs[1].get_lines()[0].get_xydata()
    assert l_speed.shape[0] == tdr.get_speed("measured").shape[0]

    fig, axs = tdr.plot(concur_vars="temperature",
                        concur_var_titles="Temperature")
    l_temp = axs[1].get_lines()[0].get_xydata()
    assert l_temp.shape[0] == tdr.get_speed("measured").shape[0]


@pytest.mark.plotting
def test_plot_zoc(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    fig, ax = tdr_calib.plot_zoc()
    lines = ax.get_lines()
    assert len(lines) >= 2
    # TODO: test arguments


@pytest.mark.plotting
def test_plot_zoc_filter(test_tdr):
    # Calibrate with "filter" and test plot_zoc. Modifies module-scoped
    # fixture.
    tdr = test_tdr.tdrX
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


@pytest.mark.plotting
def test_plot_phases(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    fig, ax = tdr_calib.plot_phases()
    lines = ax.get_lines()
    assert len(lines) == 2
    # Test selected dives
    # random dives
    rdives = np.random.choice(test_tdr_calib.diveNo_seq, 10).tolist()
    fig, ax = tdr_calib.plot_phases(diveNo=rdives)
    l_depth = ax.get_lines()[0].get_ydata()
    tdr_dives = (tdr_calib
                 .extract_dives(rdives, calib_depth=True,
                                calib_speed=False))
    assert l_depth.size == tdr_dives.depth.size

    # Test selected dives with surface
    fig, ax = tdr_calib.plot_phases(diveNo=rdives, surface=True)
    # Compare scatter collection against line data
    l_depth = ax.get_lines()[0].get_xydata()
    scat_data = ax.collections[0].get_offsets().data
    assert scat_data.size <= l_depth.size

    # Test selected dives with concur_vars
    fig, axs = tdr_calib.plot_phases(diveNo=rdives, concur_vars="speed")
    # We should have 2 Axes
    assert len(axs) == 2
    l_depth = axs[0].get_lines()[0].get_xydata()
    l_speed = axs[1].get_lines()[0].get_xydata()
    assert l_depth.size == l_speed.size
    assert l_depth.shape[0] == tdr_dives.depth.size

    # Test selected dives with concur_vars and surface
    fig, axs = tdr_calib.plot_phases(diveNo=rdives, concur_vars="speed",
                                     surface=True)
    assert len(axs) == 2
    l_depth = axs[0].get_lines()[0].get_xydata()
    l_speed = axs[1].get_lines()[0].get_xydata()
    speed_scat = axs[1].collections[0].get_offsets().data
    assert l_depth.size == l_speed.size
    assert speed_scat.size <= l_speed.size


@pytest.mark.plotting
def test_plot_dive_model(test_tdr_calib):
    tdr_calib = test_tdr_calib.tdrX
    # Test selected random dives
    rdives = np.random.choice(test_tdr_calib.diveNo_seq, 10).tolist()

    for dive in rdives:
        print("diveNo: {}".format(dive))
        fig, axs = tdr_calib.plot_dive_model(diveNo=dive)
        # TODO
        assert len(axs) == 2
        lines_ax0 = axs[0].get_lines()
        assert len(lines_ax0) == 4
        lines_ax1 = axs[1].get_lines()
        assert len(lines_ax1) == 7
