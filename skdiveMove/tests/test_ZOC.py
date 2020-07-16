"""Unit test for TDR class

"""

import unittest as ut
# import numpy.testing as npt
import xarray as xr
import skdiveMove as skdive
from skdiveMove.tests import diveMove2skd


class TestZOC(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdrX = diveMove2skd(True)  # TDRSource
        self.zocX = skdive.tdr.ZOC()

    def test_init(self):
        self.assertIsInstance(self.zocX, skdive.tdr.ZOC)
        self.assertIsNone(self.zocX.method)
        self.assertIsNone(self.zocX.params)
        self.assertIsNone(self.zocX.depth_zoc)
        self.assertIsNone(self.zocX.filters)

    def test_offset_depth(self):
        depth = self.tdrX.get_depth()
        self.zocX.offset_depth(depth, offset=3)
        depth_zoc = self.zocX.get_depth()
        self.assertIsInstance(depth_zoc, xr.DataArray)
        self.assertIn("offset", self.zocX.params)
        self.assertEqual(self.zocX.method, "offset")
        attr_hist = depth_zoc.attrs["history"]
        self.assertIn("ZOC", attr_hist)

    def test_get_depth(self):
        depth = self.tdrX.get_depth()
        self.zocX.offset_depth(depth, offset=3)
        depth_zoc = self.zocX.get_depth()
        self.assertIsInstance(depth_zoc, xr.DataArray)

    def test_get_params(self):
        depth = self.tdrX.get_depth()
        self.zocX.offset_depth(depth, offset=3)
        params = self.zocX.get_params()
        self.assertIsInstance(params, dict)

    @ut.skip("test takes too long")
    def test_filter_depth(self):
        """Test for "filter" method

        Disabled, as it is too long

        """
        depth = self.tdrX.get_depth()
        DB = [-2, 5]
        K = [3, 5760]
        P = [0.5, 0.02]
        self.zocX.filter_depth(depth, k=K, probs=P, depth_bounds=DB)
        self.assertIsInstance(self.zocX.depth_zoc, xr.DataArray)
        self.assertIn("k", self.zocX.params)
        self.assertIn("probs", self.zocX.params)
        self.assertIn("depth_bounds", self.zocX.params)
        self.assertIn("na_rm", self.zocX.params)
        self.assertEqual(self.zocX.method, "filter")

    def test_call(self):
        depth = self.tdrX.get_depth()
        self.zocX(depth, "offset", offset=3)
        self.assertIsInstance(self.zocX.depth_zoc, xr.DataArray)
        self.assertIn("offset", self.zocX.params)
        self.assertEqual(self.zocX.method, "offset")


if __name__ == '__main__':
    ut.main()
