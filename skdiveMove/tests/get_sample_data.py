"""Helper function to load sample data onto a TDR object

"""

import importlib.resources as rsrc
from skdiveMove.tdr import TDR
from skdiveMove.tdrsource import TDRSource
from skdiveMove.tdrphases import TDRPhases
from skdiveMove.zoc import ZOC
from skdiveMove.imutools import IMU2Body


def diveMove2skd(oclass="TDR", has_speed=True):
    """Create a `TDRSource` instance from `diveMove` example data set

    The sample data set has been converted to NetCDF4 format, to include
    all necessary metadata.

    Parameters
    ----------
    oclass : {"TDRSource", "ZOC", "TDRPhases", "TDR"}
        The class to return.
    has_speed : bool, optional
        Whether to set the `has_speed` attribute

    Returns
    -------
    `TDR`

    """
    class_names = ["TDRSource", "ZOC", "TDRPhases", "TDR"]

    if oclass in class_names:
        ifile = (rsrc.files("skdiveMove") / "tests" /
                 "data" / "ag_mk7_2002_022.nc")

        classes = [TDRSource, ZOC, TDRPhases, TDR]
        class_map = dict(zip(class_names, classes))
        tdrX = class_map[oclass].read_netcdf(ifile, depth_name="depth",
                                             time_name="date_time",
                                             has_speed=has_speed)
    else:
        raise LookupError(("Requested class ({}) does not exist"
                           .format(oclass)))

    return tdrX


def _nc2imu2body():
    """Get example data into a IMU2Body instance

    Convenience function for tests and examples.

    """
    ncfname = "gert_imu_frame.nc"
    icdf = (rsrc.files("skdiveMove") / "tests" /
            "data" / "gertrude" / ncfname)
    icsv = (rsrc.files("skdiveMove") / "tests" /
            "data" / "gertrude" / "gert_long_srfc.csv")

    return IMU2Body.from_csv_nc(icsv, icdf, imu_filename=ncfname)
