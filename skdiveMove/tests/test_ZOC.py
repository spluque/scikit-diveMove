"""Unit test for TDR class

"""

import unittest as ut
# import numpy.testing as npt
import xarray as xr
from skdiveMove.zoc import ZOC
from skdiveMove.tests import diveMove2skd


class TestZOC(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.zocX = diveMove2skd("ZOC", True)

    def test_init(self):
        self.assertIsInstance(self.zocX, ZOC)
        self.assertIsNone(self.zocX.zoc_method)
        self.assertIsNone(self.zocX._zoc_params)
        self.assertIsNone(self.zocX._depth_zoc)
        self.assertIsNone(self.zocX.zoc_filters)

    def test_str(self):
        self.assertIn("Class ZOC object", self.zocX.__str__())

    def test_offset_depth(self):
        self.zocX._offset_depth(offset=3)
        depth_zoc = self.zocX.depth_zoc
        self.assertIsInstance(depth_zoc, xr.DataArray)
        self.assertIn("offset", self.zocX.zoc_params)
        self.assertEqual(self.zocX.zoc_method, "offset")
        attr_hist = depth_zoc.attrs["history"]
        self.assertIn("ZOC", attr_hist)

    def test_get_depth(self):
        self.zocX.zoc("offset", offset=3)
        depth_zoc = self.zocX.depth
        self.assertIsInstance(depth_zoc, xr.DataArray)

    def test_get_params(self):
        self.zocX.zoc("offset", offset=3)
        params = self.zocX.zoc_params
        self.assertIsInstance(params, tuple)

    # @ut.skip("test takes too long")
    def test_filter_depth(self):
        """Test for "filter" method

        Test target is the process, *not* the result.  The 2nd window width
        is not what the data call for, but was chosen for test performance
        reasons.

        """
        DB = [-2, 5]
        K = [3, 600]
        P = [0.5, 0.02]
        self.zocX._filter_depth(k=K, probs=P, depth_bounds=DB)
        self.assertIsInstance(self.zocX.depth_zoc, xr.DataArray)
        self.assertIn("k", self.zocX.zoc_params[1])
        self.assertIn("probs", self.zocX.zoc_params[1])
        self.assertIn("depth_bounds", self.zocX.zoc_params[1])
        self.assertIn("na_rm", self.zocX.zoc_params[1])
        self.assertEqual(self.zocX.zoc_method, "filter")
        attr_hist = self.zocX.depth_zoc.attrs["history"]
        self.assertIn("ZOC", attr_hist)

    def test_zoc_offset(self):
        self.zocX.zoc("offset", offset=3)
        self.assertIsInstance(self.zocX.depth_zoc, xr.DataArray)
        self.assertIn("offset", self.zocX.zoc_params)
        self.assertEqual(self.zocX.zoc_method, "offset")

    def test_zoc_filter(self):
        DB = [-2, 5]
        K = [3, 600]
        P = [0.5, 0.02]
        self.zocX.zoc("filter", k=K, probs=P, depth_bounds=DB)
        self.assertIsInstance(self.zocX.depth_zoc, xr.DataArray)
        self.assertIn("filter", self.zocX.zoc_params)
        self.assertIn("k", self.zocX.zoc_params[1])
        self.assertIn("probs", self.zocX.zoc_params[1])
        self.assertIn("depth_bounds", self.zocX.zoc_params[1])
        self.assertIn("na_rm", self.zocX.zoc_params[1])
        self.assertEqual(self.zocX.zoc_method, "filter")
        attr_hist = self.zocX.depth_zoc.attrs["history"]
        self.assertIn("ZOC", attr_hist)


if __name__ == '__main__':
    ut.main()
