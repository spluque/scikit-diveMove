"""Base definition of the TDR input data

"""

import xarray as xr
from skdiveMove.helpers import get_var_sampling_interval

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
    >>> tdrX = diveMove2skd()
    >>> print(tdrX)  # doctest: +ELLIPSIS
    Time-Depth Recorder -- Class TDR object ...

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
        x = self.tdr
        xdf = x.to_dataframe()
        objcls = ("Time-Depth Recorder -- Class {} object\n"
                  .format(self.__class__.__name__))
        src = "{0:<20} {1}\n".format("Source File", self.tdr_file)
        itv = ("{0:<20} {1}\n"
               .format("Sampling interval",
                       get_var_sampling_interval(x[self.depth_name])))
        nsamples = "{0:<20} {1}\n".format("Number of Samples",
                                          xdf.shape[0])
        beg = "{0:<20} {1}\n".format("Sampling Begins",
                                     xdf.index[0])
        end = "{0:<20} {1}\n".format("Sampling Ends",
                                     xdf.index[-1])
        dur = "{0:<20} {1}\n".format("Total duration",
                                     xdf.index[-1] - xdf.index[0])
        drange = "{0:<20} [{1},{2}]\n".format("Measured depth range",
                                              xdf[self.depth_name].min(),
                                              xdf[self.depth_name].max())
        others = "{0:<20} {1}\n".format("Other variables",
                                        [x for x in xdf.columns
                                         if x != self.depth_name])
        attr_list = "Attributes:\n"
        for key, val in sorted(x.attrs.items()):
            attr_list += "{0:>35}: {1}\n".format(key, val)
        attr_list = attr_list.rstrip("\n")

        return(objcls + src + itv + nsamples + beg + end + dur + drange +
               others + attr_list)

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
