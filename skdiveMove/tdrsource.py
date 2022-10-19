"""Base definition of the TDR input data

"""

import pandas as pd
from skdiveMove.helpers import (get_var_sampling_interval,
                                _append_xr_attr, _load_dataset)

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
    time_name : str
        Name of the time dimension in the dataset.
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
    def __init__(self, dataset, depth_name="depth", time_name="timestamp",
                 subsample=None, has_speed=False, tdr_filename=None):
        """Set up attributes for TDRSource objects

        Parameters
        ----------
        dataset : xarray.Dataset
            Dataset containing depth, and optionally other DataArrays.
        depth_name : str, optional
            Name of data variable with depth measurements.
        time_name : str, optional
            Name of the time dimension in the dataset.
        subsample : str, optional
            Subsample dataset at given frequency specification.  See pandas
            offset aliases.
        has_speed : bool, optional
            Weather data includes speed measurements. Column name must be
            one of ["velocity", "speed"].
        tdr_filename : str
            Name of the file from which `dataset` originated.

        """
        self.time_name = time_name
        if subsample is not None:
            self.tdr = (dataset.resample({time_name: subsample})
                        .interpolate("linear"))
            for vname, da in self.tdr.data_vars.items():
                da.attrs["sampling_rate"] = (1.0 /
                                             pd.to_timedelta(subsample)
                                             .seconds)
                da.attrs["sampling_rate_units"] = "Hz"
                _append_xr_attr(da, "history",
                                "Resampled to {}\n".format(subsample))
        else:
            self.tdr = dataset
        self.depth_name = depth_name
        speed_var = [x for x in list(self.tdr.data_vars.keys())
                     if x in _SPEED_NAMES]
        if speed_var and has_speed:
            self.has_speed = True
            self.speed_name = speed_var[0]
        else:
            self.has_speed = False
            self.speed_name = None

        self.tdr_file = tdr_filename

    @classmethod
    def read_netcdf(cls, tdr_file, depth_name="depth", time_name="timestamp",
                    subsample=None, has_speed=False, **kwargs):
        """Instantiate object by loading Dataset from NetCDF file

        Parameters
        ----------
        tdr_file : str
            As first argument for :func:`xarray.load_dataset`.
        depth_name : str, optional
            Name of data variable with depth measurements. Default: "depth".
        time_name : str, optional
            Name of the time dimension in the dataset.
        subsample : str, optional
            Subsample dataset at given frequency specification.  See pandas
            offset aliases.
        has_speed : bool, optional
            Weather data includes speed measurements. Column name must be
            one of ["velocity", "speed"].  Default: False.
        **kwargs : optional keyword arguments
            Arguments passed to :func:`xarray.load_dataset`.

        Returns
        -------
        obj : TDRSource, ZOC, TDRPhases, or TDR
            Class matches the caller.

        """
        dataset = _load_dataset(tdr_file, **kwargs)
        return cls(dataset, depth_name=depth_name, time_name=time_name,
                   subsample=subsample, has_speed=has_speed,
                   tdr_filename=tdr_file)

    def __str__(self):
        x = self.tdr
        depth_xr = x[self.depth_name]
        depth_ser = depth_xr.to_series()
        objcls = ("Time-Depth Recorder -- Class {} object\n"
                  .format(self.__class__.__name__))
        src = "{0:<20} {1}\n".format("Source File", self.tdr_file)
        itv = ("{0:<20} {1}\n"
               .format("Sampling interval",
                       get_var_sampling_interval(depth_xr)))
        nsamples = "{0:<20} {1}\n".format("Number of Samples",
                                          depth_xr.shape[0])
        beg = "{0:<20} {1}\n".format("Sampling Begins",
                                     depth_ser.index[0])
        end = "{0:<20} {1}\n".format("Sampling Ends",
                                     depth_ser.index[-1])
        dur = "{0:<20} {1}\n".format("Total duration",
                                     depth_ser.index[-1] -
                                     depth_ser.index[0])
        drange = "{0:<20} [{1},{2}]\n".format("Measured depth range",
                                              depth_ser.min(),
                                              depth_ser.max())
        others = "{0:<20} {1}\n".format("Other variables",
                                        [x for x in list(x.keys())
                                         if x != self.depth_name])
        attr_list = "Attributes:\n"
        for key, val in sorted(x.attrs.items()):
            attr_list += "{0:>35}: {1}\n".format(key, val)
        attr_list = attr_list.rstrip("\n")

        return (objcls + src + itv + nsamples + beg + end + dur + drange +
                others + attr_list)

    def _get_depth(self):
        return self.tdr[self.depth_name]

    depth = property(_get_depth)
    """Return depth array

    Returns
    -------
    xarray.DataArray

    """

    def _get_speed(self):
        return self.tdr[self.speed_name]

    speed = property(_get_speed)
    """Return speed array

    Returns
    -------
    xarray.DataArray

    """
