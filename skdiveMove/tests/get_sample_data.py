"""Helper function to load sample data onto a TDR object

"""

import os
import skdiveMove as skdive


def diveMove2skd(isTDRSource=False):
    """Create a `TDRSource` instance from `diveMove` example data set

    The sample data set has been converted to NetCDF4 format, to include
    all necessary metadata.

    Returns
    -------
    `TDR`

    """
    myPath = os.path.dirname(os.path.abspath(__file__))
    ifile = os.path.join(myPath, "data", "ag_mk7_2002_022.nc")

    if isTDRSource:
        tdrX = skdive.tdr.TDRSource(ifile, depth_name="depth",
                                    has_speed=True)
    else:
        tdrX = skdive.TDR(ifile, depth_name="depth", has_speed=True)

    return(tdrX)
