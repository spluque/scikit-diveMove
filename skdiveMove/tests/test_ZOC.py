"""Test TDR ZOC class

"""

# import numpy.testing as npt
import xarray as xr
from skdiveMove.zoc import ZOC
from skdiveMove.tests.get_sample_data import diveMove2skd
import pytest


@pytest.fixture(scope="module")
def zocX():
    """Load a zero-offset corrected TDR instance for testing"""
    return diveMove2skd("ZOC", has_speed=True)


def test_init(zocX):
    assert isinstance(zocX, ZOC)
    assert zocX.zoc_method is None
    assert zocX._zoc_params is None
    assert zocX._depth_zoc is None
    assert zocX.zoc_filters is None


def test_str(zocX):
    assert "Class ZOC object" in zocX.__str__()


def test_offset_depth(zocX):
    zocX._offset_depth(offset=3)
    depth_zoc = zocX.depth_zoc
    assert isinstance(depth_zoc, xr.DataArray)
    assert "offset" in zocX.zoc_params
    assert zocX.zoc_method == "offset"
    attr_hist = depth_zoc.attrs["history"]
    assert "ZOC" in attr_hist


def test_get_depth(zocX):
    zocX.zoc("offset", offset=3)
    depth_zoc = zocX.depth
    assert isinstance(depth_zoc, xr.DataArray)


def test_get_params(zocX):
    zocX.zoc("offset", offset=3)
    params = zocX.zoc_params
    assert isinstance(params, tuple)


def test_filter_depth(zocX):
    """Test for "filter" method

    Test target is the process, *not* the result.  The 2nd window width
    is not what the data call for, but was chosen for test performance
    reasons.

    """
    DB = [-2, 5]
    K = [3, 600]
    P = [0.5, 0.02]
    zocX._filter_depth(k=K, probs=P, depth_bounds=DB)
    assert isinstance(zocX.depth_zoc, xr.DataArray)
    assert "k" in zocX.zoc_params[1]
    assert "probs" in zocX.zoc_params[1]
    assert "depth_bounds" in zocX.zoc_params[1]
    assert "na_rm" in zocX.zoc_params[1]
    assert zocX.zoc_method == "filter"
    attr_hist = zocX.depth_zoc.attrs["history"]
    assert "ZOC" in attr_hist


def test_zoc_offset(zocX):
    zocX.zoc("offset", offset=3)
    assert isinstance(zocX.depth_zoc, xr.DataArray)
    assert "offset" in zocX.zoc_params
    assert zocX.zoc_method == "offset"


def test_zoc_filter(zocX):
    DB = [-2, 5]
    K = [3, 600]
    P = [0.5, 0.02]
    zocX.zoc("filter", k=K, probs=P, depth_bounds=DB)
    assert isinstance(zocX.depth_zoc, xr.DataArray)
    assert "filter" in zocX.zoc_params
    assert "k" in zocX.zoc_params[1]
    assert "probs" in zocX.zoc_params[1]
    assert "depth_bounds" in zocX.zoc_params[1]
    assert "na_rm" in zocX.zoc_params[1]
    assert zocX.zoc_method == "filter"
    attr_hist = zocX.depth_zoc.attrs["history"]
    assert "ZOC" in attr_hist
