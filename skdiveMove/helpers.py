"""Utilities to help with simple and repetitive tasks

"""

import pandas as pd
from skdiveMove.core import robjs, cv, pandas2ri


def _get_dive_indices(indices, diveNo):
    """Mapping to diveMove's `.diveIndices`"""
    rstr = """diveIDXFun <- diveMove:::.diveIndices"""
    dive_idx_fun = robjs.r(rstr)
    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        # Subtract 1 for zero-based python
        idx_ok = dive_idx_fun(indices, diveNo) - 1

    return(idx_ok)


def _add_xr_attr(x, attr, val):
    """Add an attribute to xarray.DataArray or xarray.Dataset

    Parameters
    ----------
    x : xarray.DataArray or xarray.Dataset
    attr : str
        Attribute name to update or add
    val : str
        Attribute value
    """
    if attr in x.attrs:
        x.attrs[attr] += ",{}".format(val)
    else:
        x.attrs["history"] = "{}".format(val)


def get_var_sampling_interval(x):
    """Retrieve sampling interval from DataArray attributes

    Parameters
    ----------
    x : xarray.DataArray

    Returns
    -------
    pandas.Timedelta

    """
    attrs = x.attrs
    intvl = (pd.Timedelta(attrs["sampling_rate"] +
                          attrs["sampling_rate_unit"]))

    return(intvl)
