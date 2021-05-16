"""Utilities to help with simple and repetitive tasks

"""

import numpy as np
import pandas as pd
from skdiveMove.core import robjs, cv, pandas2ri

__all__ = ["_get_dive_indices", "_add_xr_attr",
           "get_var_sampling_interval", "_cut_dive",
           "_one_dive_stats", "_speed_stats", "rle_key"]


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


def _cut_dive(x, dive_model, smooth_par, knot_factor,
              descent_crit_q, ascent_crit_q):
    """Private function to retrieve results from `diveModel` object in R

    Parameters
    ----------
    x : pandas.DataFrame
        Subset with a single dive's data, with first column expected to be
        dive ID.
    dive_model : str
    smooth_par : float
    knot_factor : int
    descent_crit_q : float
    ascent_crit_q : float

    Notes
    -----
    See details for arguments in diveMove's ``calibrateDepth``.  This
    function maps to ``diveMove:::.cutDive``, and only sets some of the
    parameters from the `R` function.

    Returns
    -------
    out : dict
        Dictionary with the following keys and corresponding component:
        {'label_matrix', 'dive_spline', 'spline_deriv', 'descent_crit',
        'ascent_crit', 'descent_crit_rate', 'ascent_crit_rate'}

    """
    xx = x.iloc[:, 1:]
    rstr = """cutDiveFun <- diveMove:::.cutDive"""
    cutDiveFun = robjs.r(rstr)
    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        dmodel = cutDiveFun(xx, dive_model=dive_model,
                            smooth_par=smooth_par,
                            knot_factor=knot_factor,
                            descent_crit_q=descent_crit_q,
                            ascent_crit_q=ascent_crit_q)
        dmodel_slots = ["label.matrix", "dive.spline", "spline.deriv",
                        "descent.crit", "ascent.crit",
                        "descent.crit.rate", "ascent.crit.rate"]

        lmtx = (np.array(robjs.r.slot(dmodel, dmodel_slots[0]))
                .reshape((xx.shape[0], 2), order="F"))
        spl = robjs.r.slot(dmodel, dmodel_slots[1])
        spl_der = robjs.r.slot(dmodel, dmodel_slots[2])
        spl_der = np.column_stack((spl_der[0], spl_der[1]))
        desc_crit = robjs.r.slot(dmodel, dmodel_slots[3])[0]
        asc_crit = robjs.r.slot(dmodel, dmodel_slots[4])[0]
        desc_crit_r = robjs.r.slot(dmodel, dmodel_slots[5])[0]
        asc_crit_r = robjs.r.slot(dmodel, dmodel_slots[6])[0]
        # Replace dots with underscore for the output
        dmodel_slots = [x.replace(".", "_") for x in dmodel_slots]
        res = dict(zip(dmodel_slots,
                       [lmtx, spl, spl_der, desc_crit, asc_crit,
                        desc_crit_r, asc_crit_r]))

    return(res)


def _one_dive_stats(x, interval, has_speed=False):
    """Calculate dive statistics for a single dive's DataFrame

    Parameters
    ----------
    x : pandas.DataFrame
        First column expected to be dive ID, the rest as in `diveMove`.
    interval : float
    has_speed : bool

    Returns
    -------
    out : pandas.DataFrame

    """
    xx = x.iloc[:, 1:]
    rstr = "one_dive_stats_fun <- diveMove::oneDiveStats"
    one_dive_stats_fun = robjs.r(rstr)
    onames_speed = ["begdesc", "enddesc", "begasc", "desctim", "botttim",
                    "asctim", "divetim", "descdist", "bottdist", "ascdist",
                    "bottdep_mean", "bottdep_median", "bottdep_sd",
                    "maxdep", "desc_tdist", "desc_mean_speed",
                    "desc_angle", "bott_tdist", "bott_mean_speed",
                    "asc_tdist", "asc_mean_speed", "asc_angle"]
    onames_nospeed = onames_speed[:14]

    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        res = one_dive_stats_fun(xx, interval, has_speed)

    if has_speed:
        onames = onames_speed
    else:
        onames = onames_nospeed

    res_df = pd.DataFrame(res, columns=onames)
    for tcol in range(3):
        # This is per POSIXct convention in R
        res_df.iloc[:, tcol] = pd.to_datetime(res_df.iloc[:, tcol],
                                              unit="s")

    return(res_df)


def _speed_stats(x, vdist=None):
    """Calculate total travel distance, mean speed, and angle from speed

    Dive stats for a single segment of a dive.

    Parameters
    ----------
    x : pandas.Series
        Series with speed measurements.

    vdist : float, optional
        Vertical distance corresponding to `x`.

    Returns
    -------
    out :

    """
    rstr = "speed_stats_fun <- diveMove:::.speedStats"
    speed_stats_fun = robjs.r(rstr)

    kwargs = dict(x=x)
    if vdist is not None:
        kwargs.update(vdist=vdist)
    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        res = speed_stats_fun(**kwargs)

    return(res)


def rle_key(x):
    """Emulate a run length encoder

    Assigns a numerical sequence identifying run lengths in input Series.

    Parameters
    ----------
    x : pandas.Series
        Series with data to encode.

    Returns
    -------
    out : pandas.Series

    Examples
    --------
    >>> N = 18
    >>> color = np.repeat(list("ABCABC"), 3)
    >>> ss = pd.Series(color,
    ...                index=pd.date_range("2020-01-01", periods=N,
    ...                                    freq="10s", tz="UTC"),
    ...                dtype="category")
    >>> rle_key(ss)
    2020-01-01 00:00:00+00:00    1
    2020-01-01 00:00:10+00:00    1
    2020-01-01 00:00:20+00:00    1
    2020-01-01 00:00:30+00:00    2
    2020-01-01 00:00:40+00:00    2
    2020-01-01 00:00:50+00:00    2
    2020-01-01 00:01:00+00:00    3
    2020-01-01 00:01:10+00:00    3
    2020-01-01 00:01:20+00:00    3
    2020-01-01 00:01:30+00:00    4
    2020-01-01 00:01:40+00:00    4
    2020-01-01 00:01:50+00:00    4
    2020-01-01 00:02:00+00:00    5
    2020-01-01 00:02:10+00:00    5
    2020-01-01 00:02:20+00:00    5
    2020-01-01 00:02:30+00:00    6
    2020-01-01 00:02:40+00:00    6
    2020-01-01 00:02:50+00:00    6
    Freq: 10S, dtype: int64

    """
    xout = x.ne(x.shift()).cumsum()
    return(xout)


if __name__ == '__main__':
    N = 18
    color = np.repeat(list("ABCABC"), 3)
    ss = pd.Series(color,
                   index=pd.date_range("2020-01-01", periods=N,
                                       freq="10s", tz="UTC"),
                   dtype="category")

    xx = pd.Series(np.random.standard_normal(10))
    rle_key(xx > 0)
