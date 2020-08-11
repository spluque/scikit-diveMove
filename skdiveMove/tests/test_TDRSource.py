"""Unit test for TDR class

"""

import unittest as ut
# import numpy.testing as npt
import xarray as xr
import skdiveMove.tdrsource as tdrsrc
from skdiveMove.tests import diveMove2skd


class TestTDRSource(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd("TDRSource", True)

    def test_init(self):
        self.assertIsInstance(self.tdrX, tdrsrc.TDRSource)
        self.assertIsInstance(self.tdrX.tdr, xr.Dataset)
        self.assertTrue(self.tdrX.has_speed)

        # Test no speed
        tdr = diveMove2skd("TDRSource", False)
        self.assertFalse(tdr.has_speed)

    def test_str(self):
        self.assertIn("Class TDRSource object", self.tdrX.__str__())

    def test_get_depth(self):
        depth = self.tdrX.depth
        self.assertIsInstance(depth, xr.DataArray)

    def test_get_speed(self):
        speed = self.tdrX.speed
        self.assertIsInstance(speed, xr.DataArray)


if __name__ == '__main__':
    ut.main()
