"""Unit test for TDR class

"""

import os
import unittest as ut
import importlib.resources as rsrc
from tempfile import NamedTemporaryFile
import skdiveMove as skdive
import skdiveMove.calibconfig as calibconfig


class TestCalibrate(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        self.tdr_file = (rsrc.files("skdiveMove") / "tests" / "data" /
                         "ag_mk7_2002_022.nc")
        self.config_file = (rsrc.files("skdiveMove") / "config_examples" /
                            "ag_mk7_2002_022_config.json")

    def test_calibrate(self):
        tdr_calib = skdive.calibrate(self.tdr_file, self.config_file)
        self.assertIsInstance(tdr_calib, skdive.TDR)

    def test_config_template(self):
        conffile = NamedTemporaryFile("r+", prefix="skdiveMove_",
                                      delete=False)

        calibconfig.dump_config_template(conffile.name)
        config = calibconfig.read_config(conffile.name)
        self.assertDictEqual(config, calibconfig._DEFAULT_CONFIG)
        os.remove(conffile.name)

    def test_dump_config(self):
        conffile = NamedTemporaryFile("r+", prefix="skdiveMove_",
                                      delete=False)

        calibconfig.dump_config(conffile.name, calibconfig._DEFAULT_CONFIG)
        config = calibconfig.read_config(conffile.name)
        self.assertDictEqual(config, calibconfig._DEFAULT_CONFIG)
        os.remove(conffile.name)


if __name__ == '__main__':
    ut.main()
