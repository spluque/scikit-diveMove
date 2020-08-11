"""Unit test for TDR class

"""

import os
import unittest as ut
import pkg_resources as pkg_rsrc
from tempfile import NamedTemporaryFile
import skdiveMove as skdive
import skdiveMove.calibconfig as calibconfig


class TestCalibrate(ut.TestCase):
    """Test `TDR` class methods

    """
    def setUp(self):
        # An instance to work with
        tdrfn = "tests/data/ag_mk7_2002_022.nc"
        self.tdr_file = (pkg_rsrc
                         .resource_filename("skdiveMove", tdrfn))
        configfn = "config_examples/ag_mk7_2002_022_config.json"
        self.config_file = (pkg_rsrc
                            .resource_filename("skdiveMove", configfn))

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
