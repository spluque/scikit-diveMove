"""Test TDRSource class

"""

# import numpy.testing as npt
import xarray as xr
import skdiveMove.tdrsource as tdrsrc
import pytest

from skdiveMove.tests.get_sample_data import diveMove2skd
from skdiveMove.helpers import get_var_sampling_interval


@pytest.fixture(scope="module")
def tdrX():
    """Load a TDRSource instance for testing."""
    return diveMove2skd("TDRSource", has_speed=True)


def test_init(tdrX):
    assert isinstance(tdrX, tdrsrc.TDRSource)
    assert isinstance(tdrX.tdr, xr.Dataset)
    assert tdrX.has_speed

    # Test no speed
    tdr = diveMove2skd("TDRSource", has_speed=False)
    assert not tdr.has_speed

    # Test subsampling
    tdr = diveMove2skd("TDRSource", has_speed=True, subsample="10s")
    assert 10 == pytest.approx(get_var_sampling_interval(tdr.depth)
                               .total_seconds())


def test_str(tdrX):
    assert "Class TDRSource object" in tdrX.__str__()


def test_get_depth(tdrX):
    depth = tdrX.depth
    assert isinstance(depth, xr.DataArray)


def test_get_speed(tdrX):
    speed = tdrX.speed
    assert isinstance(speed, xr.DataArray)
