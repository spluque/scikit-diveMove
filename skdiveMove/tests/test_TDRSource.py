"""Unit test for TDR class

"""

import unittest as ut
# import numpy.testing as npt
import xarray as xr
import skdiveMove as skdive
from skdiveMove.tests import diveMove2skd


class TestTDRSource(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd(True)

    def test_init(self):
        self.assertIsInstance(self.tdrX, skdive.tdr.TDRSource)
        self.assertIsInstance(self.tdrX.tdr, xr.Dataset)
        self.assertTrue(self.tdrX.has_speed)

    def test_get_depth(self):
        depth = self.tdrX.get_depth()
        self.assertIsInstance(depth, xr.DataArray)

    def test_get_speed(self):
        speed = self.tdrX.get_speed()
        self.assertIsInstance(speed, xr.DataArray)


if __name__ == '__main__':
    ut.main()
