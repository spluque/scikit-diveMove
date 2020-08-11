"""Helper function to load sample data onto a TDR object

"""

import pkg_resources as pkg_rsrc
from skdiveMove.tdr import TDR
from skdiveMove.tdrsource import TDRSource
from skdiveMove.tdrphases import TDRPhases
from skdiveMove.zoc import ZOC


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
        ifile = (pkg_rsrc
                 .resource_filename("skdiveMove",
                                    ("tests/data/"
                                     "ag_mk7_2002_022.nc")))

        classes = [TDRSource, ZOC, TDRPhases, TDR]
        class_map = dict(zip(class_names, classes))
        tdrX = class_map[oclass](ifile, depth_name="depth",
                                 has_speed=has_speed)
    else:
        raise LookupError(("Requested class ({}) does not exist"
                           .format(oclass)))

    return(tdrX)
