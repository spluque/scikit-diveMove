"""Test calibrate method

"""

import importlib.resources as rsrc
import skdiveMove as skdive
import skdiveMove.calibconfig as calibconfig
import pytest

from pathlib import Path
from dataclasses import asdict


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
    conffile = tmp_path / "template_config.json"

    calibconfig.dump_config_template(conffile)
    config = calibconfig.ConfigTDR.from_config_file(conffile)
    config_d = asdict(config)
    del config_d["dives"]["smooth_par"]  # remove injected default
    assert config_d == calibconfig._DEFAULT_CONFIG


def test_dump_config(tdr_ifiles, tmp_path: Path):
    _, config_file = tdr_ifiles
    conffile = tmp_path / "tdr_config.json"

    config_in = calibconfig.ConfigTDR.from_config_file(config_file)
    config_in.dump_config(conffile)
    config_out = calibconfig.ConfigTDR.from_config_file(conffile)
    config_in_d = asdict(config_in)
    assert config_in_d == asdict(config_out)
