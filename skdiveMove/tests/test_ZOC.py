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
        self.assertIsNone(self.zocX._params)
        self.assertIsNone(self.zocX._depth_zoc)
        self.assertIsNone(self.zocX.filters)

    def test_offset_depth(self):
        depth = self.tdrX.depth
        self.zocX.offset_depth(depth, offset=3)
        depth_zoc = self.zocX.depth
        self.assertIsInstance(depth_zoc, xr.DataArray)
        self.assertIn("offset", self.zocX.params)
        self.assertEqual(self.zocX.method, "offset")
        attr_hist = depth_zoc.attrs["history"]
        self.assertIn("ZOC", attr_hist)

    def test_get_depth(self):
        depth = self.tdrX.depth
        self.zocX.offset_depth(depth, offset=3)
        depth_zoc = self.zocX.depth
        self.assertIsInstance(depth_zoc, xr.DataArray)

    def test_get_params(self):
        depth = self.tdrX.depth
        self.zocX.offset_depth(depth, offset=3)
        params = self.zocX.params
        self.assertIsInstance(params, tuple)

    # @ut.skip("test takes too long")
    def test_filter_depth(self):
        """Test for "filter" method

        Test target is the process, *not* the result.  The 2nd window width
        is not what the data call for, but was chosen for test performance
        reasons.

        """
        depth = self.tdrX.depth
        DB = [-2, 5]
        K = [3, 600]
        P = [0.5, 0.02]
        self.zocX.filter_depth(depth, k=K, probs=P, depth_bounds=DB)
        self.assertIsInstance(self.zocX.depth, xr.DataArray)
        self.assertIn("k", self.zocX.params[1])
        self.assertIn("probs", self.zocX.params[1])
        self.assertIn("depth_bounds", self.zocX.params[1])
        self.assertIn("na_rm", self.zocX.params[1])
        self.assertEqual(self.zocX.method, "filter")
        attr_hist = self.zocX.depth.attrs["history"]
        self.assertIn("ZOC", attr_hist)

    def test_call_offset(self):
        depth = self.tdrX.depth
        self.zocX(depth, "offset", offset=3)
        self.assertIsInstance(self.zocX.depth, xr.DataArray)
        self.assertIn("offset", self.zocX.params)
        self.assertEqual(self.zocX.method, "offset")

    def test_call_filter(self):
        depth = self.tdrX.depth
        DB = [-2, 5]
        K = [3, 600]
        P = [0.5, 0.02]
        self.zocX(depth, k=K, probs=P, depth_bounds=DB)
        self.assertIsInstance(self.zocX.depth, xr.DataArray)
        self.assertIn("filter", self.zocX.params)
        self.assertIn("k", self.zocX.params[1])
        self.assertIn("probs", self.zocX.params[1])
        self.assertIn("depth_bounds", self.zocX.params[1])
        self.assertIn("na_rm", self.zocX.params[1])
        self.assertEqual(self.zocX.method, "filter")
        attr_hist = self.zocX.depth.attrs["history"]
        self.assertIn("ZOC", attr_hist)


if __name__ == '__main__':
    ut.main()
