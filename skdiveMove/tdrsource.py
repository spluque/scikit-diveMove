"""Base definition of the TDR input data

"""

import xarray as xr

_SPEED_NAMES = ["velocity", "speed"]


class TDRSource:
    """Define TDR data source

    Use xarray.Dataset to ensure pseudo-standard metadata

    Attributes
    ----------
    tdr_file : str
        String indicating the file where the data comes from.
    tdr : xarray.Dataset
        Dataset with input data.
    depth_name : str
        Name of data variable with depth measurements.
    has_speed : bool
        Whether input data include speed measurements.
    speed_name : str
        Name of data variable with the speed measurements.

    Examples
    --------
    >>> from skdiveMove.tests import diveMove2skd
    >>> tdrX = diveMove2skd(True)
    >>> print(tdrX)

    """
    def __init__(self, tdr_file, depth_name="depth",
                 has_speed=False, **kwargs):
        """Set up attributes for TDRSource objects

        Parameters
        ----------
        tdr_file : str, Path or xarray.backends.*DataStore
            As first argument for :func:`xarray.load_dataset`.
        depth_name : str, optional
            Name of data variable with depth measurements. Default: "depth".
        has_speed : bool, optional
            Weather data includes speed measurements. Column name must be
            one of ["velocity", "speed"].  Default: False.
        **kwargs : optional keyword arguments
            Arguments passed to ``xarray.load_dataset``.

        """
        self.tdr = xr.load_dataset(tdr_file, **kwargs)
        self.depth_name = depth_name
        speed_var = [x for x in list(self.tdr.data_vars.keys())
                     if x in _SPEED_NAMES]
        if speed_var and has_speed:
            self.has_speed = True
            self.speed_name = speed_var[0]
        else:
            self.has_speed = False
            self.speed_name = None

        self.tdr_file = tdr_file

    def __str__(self):
        objcls = ("Time-Depth Recorder data -- Class {} object\n"
                  .format(self.__class__.__name__))
        return(objcls + "{}".format(self.tdr))

    def _get_depth(self):
        return(self.tdr[self.depth_name])

    depth = property(_get_depth)
    """Return depth array

    Returns
    -------
    xarray.DataArray

    """

    def _get_speed(self):
        return(self.tdr[self.speed_name])

    speed = property(_get_speed)
    """Return speed array

    Returns
    -------
    xarray.DataArray

    """
