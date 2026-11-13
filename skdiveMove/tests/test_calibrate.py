"""Test calibrate method

"""

import importlib.resources as rsrc
import skdiveMove as skdive
import skdiveMove.calibconfig as calibconfig
import pytest

from pathlib import Path


@pytest.fixture(scope="module")
def tdr_ifiles():
    """Return paths to NetCDF file and corresponding configuration"""
    tdr_file = (rsrc.files("skdiveMove") / "tests" / "data" /
                "ag_mk7_2002_022.nc")
    config_file = (rsrc.files("skdiveMove") / "config_examples" /
                   "ag_mk7_2002_022_config.json")
    return tdr_file, config_file


def test_calibrate(tdr_ifiles):
    tdr_file, config_file = tdr_ifiles
    tdr_calib = skdive.calibrate(tdr_file, config_file)
    assert isinstance(tdr_calib, skdive.TDR)


def test_config_template(tmp_path: Path):
    conffile = tmp_path / "config.json"

    calibconfig.dump_config_template(conffile)
    config = calibconfig.read_config(conffile)
    assert config == calibconfig._DEFAULT_CONFIG


def test_dump_config(tmp_path: Path):
    conffile = tmp_path / "config.json"

    calibconfig.dump_config(conffile, calibconfig._DEFAULT_CONFIG)
    config = calibconfig.read_config(conffile)
    assert config == calibconfig._DEFAULT_CONFIG
