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
        tdr_file : str, path object of file-like object
            A valid string path for the file with TDR measurements, or path
            or file-like object, interpreted as in ``pandas.read_csv``.
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

    def get_depth(self):
        """Return depth array

        Returns
        -------
        xarray.DataArray

        """
        return(self.tdr[self.depth_name])

    def get_speed(self):
        """Return speed array

        Returns
        -------
        xarray.DataArray

        """
        return(self.tdr[self.speed_name])

    def get_tdr(self):
        """Return TDR Dataset

        Returns
        -------
        xarray.Dataset

        """
        return(self.tdr)
