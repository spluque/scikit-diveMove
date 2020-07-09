"""TDR objects homologous to R package diveMove's main classes

The :class:`TDR` aims to be a comprehensive class to encapsulate the
processing of `TDR` records from a data file.

This module instantiates an ``R`` session to interact with low-level
functions and methods of package ``diveMove``.

Class & Methods Summary
-----------------------

.. autosummary::

   TDR
   TDR.calibrate
   TDR.zoc
   TDR.detect_wet
   TDR.detect_dives
   TDR.detect_dive_phases
   TDR.plot
   TDR.plot_phases
   TDR.plotZOCfilters

API
---

"""

import logging
import numpy as np
import pandas as pd
from rpy2.robjects.packages import importr
import rpy2.robjects as robjs
import rpy2.robjects.conversion as cv
from rpy2.robjects import pandas2ri
import skdiveMove.plotting as plotting
import skdiveMove.calibrate_speed as speedcal


logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())

_SPEED_NAMES = ["velocity", "speed"]

# Initialize R instance
diveMove = importr("diveMove")


def _depth_filter_r(depth, k, probs, depth_bounds, na_rm=True):
    """Filter method for zero offset correction via diveMove

    Parameters
    ----------
    depth : pandas.Series
    k : array_like
    probs : array_like
    depth_bounds : array_like
    na_rm : bool, optional

    Returns
    -------
    out : pandas.DataFrame
        Time-indexed DataFrame with a column for each filter applied, and a
        column `depth_adj` for corrected depth.

    """
    filterFun = robjs.r("""filterFun <- diveMove:::.depthFilter""")
    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        depthmtx = filterFun(depth, pd.Series(k), pd.Series(probs),
                             pd.Series(depth_bounds), na_rm)

    colnames = ["k{0}_p{1}".format(k, p) for k, p in zip(k, probs)]
    colnames.append("depth_adj")
    return(pd.DataFrame(depthmtx, index=depth.index, columns=colnames))


def _depth_filter(depth, k, probs, depth_bounds, na_rm=True):
    """Filter method for zero offset correction using Pandas

    Parameters
    ----------
    depth : pandas.Series
    k : array_like
    probs : array_like
    depth_bounds : array_like
    na_rm : bool, optional

    Notes
    -----
    This doesn't work exactly like R's version, as it uses Pandas rolling
    quantile funtion.

    TODO: find a way to do this with signal filters (e.g. `scipy.signal`).

    """
    isna_depth = depth.isna()
    isin_bounds = (depth > depth_bounds[0]) & (depth < depth_bounds[1])
    if na_rm:
        isok_depth = ~isna_depth & isin_bounds
    else:
        isok_depth = isin_bounds | isna_depth

    filters = pd.DataFrame({'depth_0': depth}, index=depth.index)
    for i, wwidth in enumerate(k):
        wname = "k{0}_p{1}".format(wwidth, probs[i])
        filters[wname] = filters.iloc[:, -1]
        dd = (filters.iloc[:, i][isok_depth]
              .rolling(wwidth, min_periods=1)
              .quantile(probs[i]))
        filters.iloc[:, i + 1][isok_depth] = dd
        # Linear interpolation for depths out of bounds
        d_intp_offbounds = (filters.iloc[:, i + 1]
                            .mask(~isin_bounds)
                            .interpolate())
        filters.iloc[:, i + 1] = d_intp_offbounds
        filters.loc[:, wname][isna_depth] = np.NAN

    filters["depth_adj"] = depth - filters[wname]
    return(filters.iloc[:, 1:])


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
    function maps to diveMove:::.cutDive, and only sets some of the
    parameters from the R function.

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

        lmtx = (robjs.r.slot(dmodel, dmodel_slots[0])
                .reshape((xx.shape[0], 2), order="F"))
        spl = robjs.r.slot(dmodel, dmodel_slots[1])
        spl_der = robjs.r.slot(dmodel, dmodel_slots[2])
        spl_der = np.column_stack((spl_der[0], spl_der[1]))
        desc_crit = robjs.r.slot(dmodel, dmodel_slots[3])[0]
        asc_crit = robjs.r.slot(dmodel, dmodel_slots[4])[0]
        desc_crit_r = robjs.r.slot(dmodel, dmodel_slots[5])[0]
        asc_crit_r = robjs.r.slot(dmodel, dmodel_slots[6])[0]
        res = dict(zip(dmodel_slots,
                       [lmtx, spl, spl_der, desc_crit, asc_crit,
                        desc_crit_r, asc_crit_r]))

    return(res)


def _one_dive_stats(x, interval, has_speed=False):
    """Calculate dive statistics for a single dive's DataFrame

    Parameters
    ----------
    x : pandas.DataFrame
        First column expected to be dive ID, the rest as in diveMove.
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


def _get_dive_indices(indices, diveNo):
    """Mapping to diveMove's `.diveIndices`"""
    rstr = """diveIDXFun <- diveMove:::.diveIndices"""
    dive_idx_fun = robjs.r(rstr)
    with cv.localconverter(robjs.default_converter +
                           pandas2ri.converter):
        # Subtract 1 for zero-based python
        idx_ok = dive_idx_fun(indices, diveNo) - 1

    return(idx_ok)


def get_diveMove_sample_data():
    """Create at `TDR` instance from diveMove example data set

    Returns
    -------
    `TDR`

    """
    rstr = ("""system.file(file.path("data", "dives.csv"), """
            """package="diveMove", mustWork=TRUE)""")
    data_path = robjs.r(rstr)[0]
    tdrX = TDR(data_path, sep=";", compression="bz2", has_speed=True)

    return(tdrX)


class TDR:
    """Base class encapsulating TDR objects and processing

    Attributes
    ----------
    src_file : str
        String indicating the file where the data comes from.
    tdr : pandas.DataFrame
        DataFrame with input data.
    dtime : str
        String representing the sampling frequency.
    has_speed : bool
        Whether input data include speed measurements.
    speed_colname : str
        Name of the speed column.
    speed_calib_fit : statsmodels fit
        The quantile regression fit.
    zoc_pars : dict
        Dictionary of ZOC parameters {'method': str, 'depth_zoc':
        pandas.Series, 'filters': pandas.DataFrame}.
    wet_act : dict
        Dictionary of wet activity data {'phases': pandas.DataFrame,
        'times_begend': pandas.DataFrame, 'dry_thr': float, 'wet_thr':
        float}.
    dives : dict
        Dictionary of dive activity data {'row_ids': pandas.DataFrame,
        'model': str, 'splines': dict, 'crit_vals': pandas.DataFrame}.

    Examples
    --------
    Construct an instance from diveMove example dataset

    >>> rstr = ('system.file(file.path("data", "dives.csv"), '
    ...         'package="diveMove", mustWork=TRUE)')
    >>> data_path = robjs.r(rstr)[0]
    >>> tdrX = TDR(data_path, sep=";", compression="bz2")

    For convenience, the above operation is wrapped in the function
    `get_diveMove_sample_data`.

    Plot the `TDR` object

    >>> tdrX.plot()

    """

    def __init__(self, tdr_file, datetime_col=[0, 1], depth_colname="depth",
                 subsample="5s", dtformat="%d/%m/%Y %H:%M:%S", utc=True,
                 has_speed=False, **kwargs):
        """Set up attributes for TDR objects

        Parameters
        ----------
        tdr_file : str, path object of file-like object
            A valid string path for the file with TDR measurements, or path
            or file-like object, interpreted as in ``pandas.read_csv``.
        datetime_col : list, optional
            List of column number(s) (zero-based) with date and time.
            Default is [0, 1].
        depth_col : int, optional
            Column number with depth measurements. Default: 2.
        subsample : DateOffset, Timedelta or str, optional
            Subsample measurements at this interval. Default: "5s".
        dtformat : str, optional
            String specifying the format in which the date and time
            columns, when pasted together, should be interpreted
        utc : bool, optional
            Whether to set the time index to "UTC".  Default: True.
        has_speed : bool, optional
            Weather data includes speed measurements. Column name must be
            one of ["velocity", "speed"].  Default: False.
        **kwargs : optional keyword arguments
            Arguments passed to ``pandas.read_csv``.

        """
        def dtparser(x):
            """Helper function for parsing date/time"""
            return(pd.to_datetime(x, format=dtformat, utc=utc))

        tdr_in = pd.read_csv(tdr_file,
                             parse_dates={'date_time': datetime_col},
                             date_parser=dtparser, index_col="date_time",
                             **kwargs)
        tdr_in.sort_index(inplace=True)
        tdr_isdups = tdr_in.index.duplicated()
        if any(tdr_isdups):
            logger.warning("There are duplicated records - {}".
                           format(tdr_in.to_string()))
        tdr_in.rename(columns={depth_colname: "depth"},
                      inplace=True)
        tdr_out = tdr_in.resample(subsample).bfill()
        self.src_file = tdr_file
        self.tdr = tdr_out
        self.dtime = subsample
        speed_col = [x for x in tdr_out.columns if x in _SPEED_NAMES]
        if speed_col and has_speed:
            self.has_speed = True
            self.speed_colname = speed_col[0]
        else:
            self.has_speed = False
            self.speed_colname = None

        # Attributes to be set from other methods: method used, corrected
        # depth, filtered depth when method="filter"
        self.zoc_pars = {'method': None,
                         'depth_zoc': None,
                         'filters': None}
        # Speed calibration fit
        self.speed_calib_fit = None
        # Wet phase activity identification
        self.wet_act = {'phases': None,
                        'times_begend': None,
                        'dry_thr': None,
                        'wet_thr': None}
        self.dives = {'row_ids': None,
                      'model': None,
                      'splines': None,
                      'spline_derivs': None,
                      'crit_vals': None}

    def __repr__(self):
        objclass = ("Time-Depth Recorder data -- Class {} object\n"
                    .format(self.__class__.__name__))
        src = "{0:<20} {1}\n".format("Source File", self.src_file)
        itv = "{0:<20} {1}\n".format("Sampling interval", self.dtime)
        nsamples = "{0:<20} {1}\n".format("Number of Samples",
                                          self.tdr.shape[0])
        beg = "{0:<20} {1}\n".format("Sampling Begins",
                                     self.tdr.index[0])
        end = "{0:<20} {1}\n".format("Sampling Ends",
                                     self.tdr.index[-1])
        dur = "{0:<20} {1}\n".format("Total duration",
                                     self.tdr.index[-1] - self.tdr.index[0])
        drange = "{0:<20} [{1},{2}]\n".format("Measured depth range",
                                              self.tdr["depth"].min(),
                                              self.tdr["depth"].max())
        others = "{0:<20} {1}\n".format("Other variables",
                                        [x for x in self.tdr.columns
                                         if x != "depth"])
        zocm = "{0:<20} \'{1}\'".format("ZOC method",
                                        self.zoc_pars["method"])
        return(objclass + src + itv + nsamples + beg + end + dur +
               drange + others + zocm)

    def zoc(self, method="filter", **kwargs):
        """Zero offset correction

        Set the ``zoc_pars`` attribute.

        Parameters
        ----------
        method : {"filter", "offset"}
            Name of method to use for zero offset correction.
        **kwargs : optional keyword arguments
            - methods 'filter': ('k', 'probs', 'depth_bounds' (defaults to
              range), 'na_rm' (defaults to True)).
            - method 'offset': ('offset').

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        Examples
        --------
        ZOC using the "offset" method

        >>> tdrX = get_diveMove_sample_data()
        >>> tdrX.zoc("offset", offset=3)

        Using the "filter" method

        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.zoc(k=K, probs=P, depth_bounds=DB)

        Plot the filters that were applied

        >>> tdrX.plotZOCfilters(ylim=[-1, 10])

        """
        depth = self.get_depth("measured")

        if (method == "offset"):
            offset = kwargs.pop("offset", 0)
            depth_zoc = depth - offset
            depth_zoc[depth_zoc < 0] = 0
            depth_zoc, zoc_filters = (depth_zoc, None)
            # Differentiate for consistency with "filter"
            depth_zoc.rename("depth_adj", inplace=True)
        elif (method == "filter"):
            k = kwargs.pop("k")         # must exist
            probs = kwargs.pop("probs")  # must exist
            # Default depth bounds equal measured depth range
            depth_bounds = kwargs.pop("depth_bounds",
                                      [depth.min(), depth.max()])
            # default as in `_depth_filter`
            na_rm = kwargs.pop("na_rm", True)
            depthmtx = _depth_filter_r(depth, k=k, probs=probs,
                                       depth_bounds=depth_bounds,
                                       na_rm=na_rm)
            depth_zoc = depthmtx.iloc[:, -1]
            depth_zoc[depth_zoc < 0] = 0
            depth_zoc, zoc_filters = (depth_zoc, depthmtx)
        else:
            logger.warning("Method {} is not implemented"
                           .format(method))

        self.zoc_pars["method"] = method
        self.zoc_pars["depth_zoc"] = depth_zoc
        self.zoc_pars["filters"] = zoc_filters

    def detect_wet(self, dry_thr=70, wet_cond=None, wet_thr=3610,
                   interp_wet=False):
        """Detect wet activity phases

        Set the ``wet_act`` attribute.

        Parameters
        ----------
        dry_thr : float, optional
        wet_cond : bool mask, optional
        wet_thr : float, optional
        interp_wet : bool, optional

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> tdrX = get_diveMove_sample_data()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases

        >>> tdrX.detect_wet()

        Access the "phases" and "times_begend" attributes

        >>> tdrX.get_wet_activity("phases")
        >>> tdrX.get_wet_activity("times_begend")

        """
        time_py = self.tdr.index
        depth_py = self.get_depth("zoc")
        dtime = pd.Timedelta(self.dtime).total_seconds()
        if wet_cond:
            wet_cond = (pd.Series(wet_cond, index=self.tdr.index)
                        .astype("bool"))
        else:
            wet_cond = ~depth_py.isna()

        rstr = """detPhaseFun <- diveMove:::.detPhase"""
        detPhaseFun = robjs.r(rstr)
        with cv.localconverter(robjs.default_converter +
                               pandas2ri.converter):
            phases_l = detPhaseFun(pd.Series(time_py), pd.Series(depth_py),
                                   dry_thr=dry_thr, wet_thr=wet_thr,
                                   wet_cond=wet_cond, interval=dtime)
            phases = pd.DataFrame({'phase_id': phases_l[0],
                                   'phase_label': phases_l[1]},
                                  index=self.tdr.index)
            phases_begend = pd.DataFrame({'beg': phases_l[2],
                                          'end': phases_l[3]})

        if interp_wet:
            zdepth = depth_py.copy()
            iswet = phases["phase_label"] == "W"
            iswetna = iswet & zdepth.isna()
            if any(iswetna):
                depth_intp = depth_py[iswet].interpolate(method="cubic")
                zdepth[iswetna] = np.maximum(np.zeros_like(depth_intp),
                                             depth_intp)
                self.zoc_pars["depth_zoc"] = zdepth

        phases.loc[:, "phase_id"] = phases.loc[:, "phase_id"].astype(int)
        self.wet_act["phases"] = phases
        self.wet_act["times_begend"] = phases_begend
        self.wet_act["dry_thr"] = dry_thr
        self.wet_act["wet_thr"] = wet_thr

    def detect_dives(self, dive_thr):
        """Identify dive events

        Set the ``dives`` attribute's "row_ids" dictionary element, and
        update the ``wet_act`` attribute's "phases" dictionary element.

        Parameters
        ----------
        dry_thr : float

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> tdrX = get_diveMove_sample_data()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        """
        depth = self.get_depth("zoc")
        act_phases = self.get_wet_activity("phases", columns="phase_label")
        detDiveFun = robjs.r("""detDiveFun <- diveMove:::.detDive""")
        with cv.localconverter(robjs.default_converter +
                               pandas2ri.converter):
            phases_df = detDiveFun(pd.Series(depth), pd.Series(act_phases),
                                   dive_thr=dive_thr)

        phases_df.set_index(depth.index, inplace=True)
        dive_activity = phases_df.pop("dive.activity")
        # Dive and post-dive ID should be integer
        phases_df = phases_df.astype(int)
        self.dives["row_ids"] = phases_df
        self.wet_act["phases"]["phase_label"] = dive_activity

    def detect_dive_phases(self, dive_model, smooth_par=0.1, knot_factor=3,
                           descent_crit_q=0, ascent_crit_q=0):
        """Detect dive phases

        Complete filling the ``dives`` attribute.

        Parameters
        ----------
        dive_model : {"unimodal", "smooth.spline"}
        smooth_par : float, optional
        knot_factor : int, optional
        descent_crit_q : float, optional
        ascent_crit_q : float, optional

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> tdrX = get_diveMove_sample_data()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        Detect dive phases using the "unimodal" method and selected
        parameters

        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)

        """
        phases_df = self.get_dive_details("row_ids")
        dive_ids = self.get_dive_details("row_ids", columns="dive.id")
        depth = self.get_depth("zoc")
        ok = (dive_ids > 0) & ~depth.isna()

        if any(ok):
            ddepths = depth[ok]  # diving depths
            dtimes = ddepths.index
            dids = dive_ids[ok]
            idx = np.squeeze(np.argwhere(ok.to_numpy()))
            time_num = (dtimes - dtimes[0]).total_seconds().to_numpy()
            divedf = pd.DataFrame({'dive_id': dids.to_numpy(),
                                   'idx': idx,
                                   'depth': ddepths.to_numpy(),
                                   'time_num': time_num},
                                  index=ddepths.index)
            grouped = divedf.groupby("dive_id")

            xx = pd.Categorical(np.repeat(["X"], phases_df.shape[0]),
                                categories=["D", "DB", "B", "BA",
                                            "DA", "A", "X"])
            self.dives["row_ids"]["dive.phase"] = xx
            dive_phases = self.dives["row_ids"]["dive.phase"]
            cval_list = []
            spl_der_list = []
            spl_list = []
            for name, grp in grouped:
                res = _cut_dive(grp, dive_model=dive_model,
                                smooth_par=smooth_par,
                                knot_factor=knot_factor,
                                descent_crit_q=descent_crit_q,
                                ascent_crit_q=ascent_crit_q)
                dive_phases.loc[grp.index] = (res.pop("label.matrix")[:, 1])
                # Splines
                spl = res.pop("dive.spline")
                # Convert directly into a dict, with each element turned
                # into a list of R objects.  Access each via
                # `_get_dive_spline_slot`
                spl_dict = dict(zip(spl.names, list(spl)))
                spl_list.append(spl_dict)
                # Spline derivatives
                spl_der = res.pop("spline.deriv")
                spl_der_idx = pd.TimedeltaIndex(spl_der[:, 0], unit="s")
                spl_der = pd.DataFrame({'y': spl_der[:, 1]},
                                       index=spl_der_idx)
                spl_der_list.append(spl_der)
                # Critical values (all that's left in res)
                cvals = pd.DataFrame(res, index=[name])
                cvals.index.rename("dive_id", inplace=True)
                # Adjust critical indices for Python convention and ensure
                # integers
                cvals.iloc[:, :2] = cvals.iloc[:, :2].astype(int) - 1
                cval_list.append(cvals)

            self.dives["model"] = dive_model
            # Splines
            self.dives["splines"] = dict(zip(grouped.groups.keys(),
                                             spl_list))
            self.dives["spline_derivs"] = pd.concat(spl_der_list,
                                                    keys=(grouped
                                                          .groups.keys()))
            self.dives["crit_vals"] = pd.concat(cval_list)

        else:
            logger.warning("No dives found")

    def calibrate(self, zoc_method="filter", dry_thr=70, wet_cond=None,
                  wet_thr=3610, interp_wet=False, dive_thr=4,
                  dive_model="unimodal", smooth_par=0.1, knot_factor=3,
                  descent_crit_q=0, ascent_crit_q=0, **kwargs):
        """Calibrate TDR object

        Convenience method to set all instance attributes.

        Parameters
        ----------
        zoc_method : {"filter", "offset"}, optional
            Name of method to use for zero offset correction.
        dry_thr : float, optional
        wet_cond : bool mask, optional
        wet_thr : float, optional
        dive_thr : float, optional
        dive_model : {"unimodal", "smooth.spline"}, optional
        smooth_par : float, optional
        knot_factor : int, optional
        descent_crit_q, ascent_crit_q : float, optional
        **kwargs : optional keyword arguments passed to ``zoc`` method
            - methods 'filter': ('k', 'probs', 'depth_bounds' (defaults to
              range), 'na_rm' (defaults to True)).
            - method 'offset': ('offset').

        Notes
        -----
        This method is homologous to diveMove's ``calibrateDepth`` function.

        Examples
        --------
        ZOC using the "filter" method

        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.calibrate(dive_thr=3, zoc_method="filter",
        ... k=K, probs=P, depth_bounds=DB, descent_crit_q=0.01,
        ... knot_factor=20)

        Plot dive phases

        >>> tdrX.plot_phases()

        Plot dive model for a dive

        >>> tdrX.plot_dive_model(40)

        """
        self.zoc(zoc_method, **kwargs)
        self.detect_wet(dry_thr=dry_thr, wet_cond=wet_cond, wet_thr=wet_thr,
                        interp_wet=interp_wet)
        self.detect_dives(dive_thr)
        self.detect_dive_phases(dive_model=dive_model,
                                smooth_par=smooth_par,
                                knot_factor=knot_factor,
                                descent_crit_q=descent_crit_q,
                                ascent_crit_q=ascent_crit_q)

    def calibrate_speed(self, tau=0.1, contour_level=0.1, z=0, bad=[0, 0],
                        save_fig=False, fname=None, **kwargs):
        """Calibrate speed measurements

        Set the `speed_calib_fit` attribute

        Parameters
        ----------
        tau : float, optional
            Quantile on which to regress speed on rate of depth change.
        contour_level : float, optional
            The mesh obtained from the bivariate kernel density estimation
            corresponding to this contour will be used for the quantile
            regression to define the calibration line.
        z : float, optional
            Only changes in depth larger than this value will be used for
            calibration.
        bad : array_like, optional
            Two-element `array_like` indicating that only rates of depth
            change and speed greater than the given value should be used
            for calibration, respectively.
        save_fig : bool, optional
            Whether to save the plot.
        fname : str, optional
            A path to save plot.  Ignored if ``save_fig=False``.
        **kwargs : optional keyword arguments
            Passed to ``calibrate_speed.calibrate``

        """
        depth = self.get_depth("zoc")
        ddiffs = depth.reset_index().diff().set_index(depth.index)
        ddepth = ddiffs["depth"].abs()
        rddepth = ddepth / ddiffs["date_time"].dt.total_seconds()
        curspeed = self.get_speed("measured")
        ok = (ddepth > z) & (rddepth > bad[0]) & (curspeed > bad[1])
        rddepth = rddepth[ok]
        curspeed = curspeed[ok]

        kde_data = pd.concat((rddepth.rename("depth_rate"),
                              curspeed), axis=1)
        qfit, fig, ax = speedcal.calibrate(kde_data, tau=tau,
                                           contour_level=contour_level,
                                           z=z, bad=bad, **kwargs)
        self.speed_calib_fit = qfit

        if save_fig:
            fig.savefig(fname)

        # ksmooth = importr("KernSmooth")
        # grdev = importr("grDevices")
        # bw_nrd = robjs.r["bw.nrd"]
        # with cv.localconverter(robjs.default_converter +
        #                        robjs.numpy2ri.converter):
        #     rddepth_np = rddepth.to_numpy()
        #     curspeed_np = curspeed.to_numpy()
        #     bandw = np.array([bw_nrd(rddepth_np),
        #                       bw_nrd(curspeed_np)])
        #     z = ksmooth.bkde2D(np.column_stack((rddepth_np,
        #                                         curspeed_np)),
        #                        bandwidth=bandw)
        #     zz = dict(zip(z.names, list(z)))
        #     bins = grdev.contourLines(zz["x1"], zz["x2"], zz["fhat"],
        #                               levels=contour_level)
        #     bins_l = list(bins)[0]
        #     bins_d = dict(zip(bins_l.names, list(bins_l)))

    def dive_stats(self, depth_deriv=True):
        """Calculate dive statistics in `TDR` records

        Parameters
        ----------
        depth_deriv : bool, optional
            Whether to compute depth derivative statistics.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        This method homologous to diveMove's `diveStats` function.

        """
        phases_df = self.get_dive_details("row_ids")
        depth = self.get_depth("zoc")

        if self.speed_calib_fit:
            speed = self.get_speed("calibrated")
        else:
            speed = self.get_speed("measured")

        dive_ids = phases_df.loc[:, "dive.id"]
        postdive_ids = phases_df.loc[:, "postdive.id"]
        ok = (dive_ids > 0) & dive_ids.isin(postdive_ids)
        okpd = (postdive_ids > 0) & postdive_ids.isin(dive_ids)

        postdive_dur = (postdive_ids[okpd].reset_index()
                        .groupby("postdive.id")
                        .apply(lambda x: x.iloc[-1] - x.iloc[0]))

        tdf = (pd.concat((phases_df[["dive.id", "dive.phase"]][ok],
                          depth[ok], speed[ok]), axis=1)
               .reset_index()
               [["dive.id", "dive.phase", "date_time",
                 "depth", self.speed_colname]])
        tdf_grp = tdf.groupby("dive.id")

        ones_list = []
        intvl = pd.Timedelta(self.dtime).total_seconds()
        for name, grp in tdf_grp:
            # Speed to be enabled once calibration is implemented
            res = _one_dive_stats(grp, interval=intvl,
                                  has_speed=self.has_speed)
            ones_list.append(res)

        ones_df = pd.concat(ones_list, ignore_index=True)
        ones_df.set_index(dive_ids[ok].unique(), inplace=True)
        ones_df.index.rename("dive_id", inplace=True)
        ones_df["postdive_dur"] = postdive_dur["date_time"]

        return(ones_df)

    def plot(self, concur_vars=None, concur_var_titles=None, **kwargs):
        """Plot TDR object

        Parameters
        ----------
        concur_vars : str or list, optional
            String or list of strings with names of columns in input to
            select additional data to plot.
        concur_var_titles : str or list, optional
            String or list of strings with y-axis labels for `concur_vars`.
        **kwargs : optional keyword arguments
            Arguments passed to ``plotTDR``.

        """
        try:
            depth = self.get_depth("zoc")
        except IndexError:
            depth = self.get_depth("measured")

        if concur_vars is None:
            plotting.plotTDR(depth, **kwargs)
        elif concur_var_titles is None:
            plotting.plotTDR(depth, concur_vars=self.tdr[concur_vars],
                             **kwargs)
        else:
            plotting.plotTDR(depth,
                             concur_vars=self.tdr.loc[:, concur_vars],
                             concur_var_titles=concur_var_titles,
                             **kwargs)

    def plotZOCfilters(self, xlim=None, ylim=None, ylab="Depth [m]"):
        """Plot zero offset correction filters

        Parameters
        ----------
        xlim, ylim : 2-tuple/list, optional
            Minimum and maximum limits for ``x``- and ``y``-axis,
            respectively.
        ylab : str, optional
            Label for ``y`` axis.

        """
        if self.zoc_pars["method"] == "filter":
            depth = self.get_depth("measured")
            zoc_filters = self.zoc_pars["filters"]
            plotting._plotZOCfilters(depth, zoc_filters, xlim, ylim, ylab)

    def plot_phases(self, diveNo=None, concur_vars=None,
                    concur_var_titles=None, surface=False, **kwargs):
        """Plot major phases found on the object

        Parameters
        ----------
        diveNo : array_like, optional
            List of dive numbers (1-based) to plot.
        concur_vars : str or list, optional
            String or list of strings with names of columns in input to
            select additional data to plot.
        concur_var_titles : str or list, optional
            String or list of strings with y-axis labels for `concur_vars`.
        **kwargs : optional keyword arguments
            Arguments passed to ``plotTDR``.

        Notes
        -----
        This is not fully functional yet:

          - dry_time is not properly handled and plotted.

        """
        row_ids = self.get_dive_details("row_ids")
        dive_ids = row_ids["dive.id"]
        dive_ids_uniq = dive_ids.unique()
        postdive_ids = row_ids["postdive.id"]

        if diveNo is None:
            diveNo = np.arange(1, row_ids["dive.id"].max() + 1).tolist()
        else:
            diveNo = [x for x in sorted(diveNo) if x in dive_ids_uniq]

        depth_all = self.get_depth("zoc").to_frame()  # need a DataFrame

        if concur_vars is None:
            dives_all = depth_all
        else:
            concur_df = self.tdr.loc[:, concur_vars]
            dives_all = pd.concat((depth_all, concur_df), axis=1)

        isin_dive_ids = dive_ids.isin(diveNo)
        isin_postdive_ids = postdive_ids.isin(diveNo)

        if surface:
            isin = isin_dive_ids | isin_postdive_ids
            dives_in = dives_all[isin]
            sfce0_idx = (postdive_ids[postdive_ids == diveNo[0] - 1]
                         .last_valid_index())
            dives_df = pd.concat((dives_all.loc[[sfce0_idx]], dives_in),
                                 axis=0)
            details_df = pd.concat((row_ids.loc[[sfce0_idx]], row_ids[isin]),
                                   axis=0)
        else:
            idx_ok = _get_dive_indices(dive_ids, diveNo)
            dives_df = dives_all.iloc[idx_ok, :]
            details_df = row_ids.iloc[idx_ok, :]

        wet_all = self.get_wet_activity("phases")
        wetdry_labs = wet_all.groupby("phase_id").nth(1)
        # Set index to integer and subtract one to match times_begend
        wetdry_labs.set_index(wetdry_labs.index.astype(int) - 1,
                              inplace=True)
        wetdry_times = self.get_wet_activity("times_begend")
        wet_dry = wetdry_times.merge(wetdry_labs, left_index=True,
                                     right_index=True)
        drys = wet_dry[wet_dry["phase_label"] == "L"]
        if (drys.shape[0] > 0):
            dry_time = drys
        else:
            dry_time = None

        if concur_vars is None:
            plotting.plotTDR(dives_df.iloc[:, 0],
                             phase_cat=details_df["dive.phase"],
                             dry_time=dry_time, **kwargs)
        else:
            plotting.plotTDR(dives_df.iloc[:, 0],
                             concur_vars=dives_df.iloc[:, 1:],
                             concur_var_titles=concur_var_titles,
                             phase_cat=details_df["dive.phase"],
                             dry_time=dry_time, **kwargs)

    def plot_dive_model(self, diveNo=None, **kwargs):
        """Plot dive model for selected dive

        Parameters
        ----------
        diveNo : array_like, optional
            List of dive numbers (1-based) to plot.
        **kwargs : optional keyword arguments

        """
        dive_ids = self.get_dive_details("row_ids", "dive.id")
        crit_vals = self.get_dive_details("crit_vals").loc[diveNo]
        idxs = _get_dive_indices(dive_ids, diveNo)
        depth = self.get_depth("zoc").iloc[idxs]
        depth_s = self._get_dive_spline_slot(diveNo, "xy")
        depth_deriv = self.get_dive_details("spline_derivs").loc[diveNo]

        # Index with time stamp
        if depth.shape[0] < 4:
            depth_s_idx = pd.date_range(depth.index[0], depth.index[-1],
                                        periods=depth_s.shape[0],
                                        tz=depth.index.tz)
            depth_s = pd.Series(depth_s.to_numpy(), index=depth_s_idx)
            dderiv_idx = pd.date_range(depth.index[0], depth.index[-1],
                                       periods=depth_deriv.shape[0],
                                       tz=depth.index.tz)
            # Extract only the series and index with time stamp
            depth_deriv = pd.Series(depth_deriv["y"].to_numpy(),
                                    index=dderiv_idx)
        else:
            depth_s = pd.Series(depth_s.to_numpy(),
                                index=depth.index[0] + depth_s.index)
            # Extract only the series and index with time stamp
            depth_deriv = pd.Series(depth_deriv["y"].to_numpy(),
                                    index=depth.index[0] + depth_deriv.index)

        # Force integer again as `loc` coerced to float above
        d_crit = crit_vals["descent.crit"].astype(int)
        a_crit = crit_vals["ascent.crit"].astype(int)
        d_crit_rate = crit_vals["descent.crit.rate"]
        a_crit_rate = crit_vals["ascent.crit.rate"]
        title = "Dive: {:d}".format(diveNo)
        plotting.plot_dive_model(depth, depth_s=depth_s,
                                 depth_deriv=depth_deriv,
                                 d_crit=d_crit, a_crit=a_crit,
                                 d_crit_rate=d_crit_rate,
                                 a_crit_rate=a_crit_rate,
                                 leg_title=title, **kwargs)

    def get_wet_activity(self, key, columns=None):
        """Accessor for the ``wet_act`` attribute

        Parameters
        ----------
        key : {"phases", "times_begend", "dry_thr", "wet_thr"}
            Name of the key to retrieve.
        columns : array_like, optional
            Names of the columns of the "phases" or times_begend dataframe.

        """
        try:
            okey = self.wet_act[key]
        except KeyError:
            msg = ("\'{}\' is not found.\nAvailable keys: {}"
                   .format(key, self.wet_act.keys()))
            logger.error(msg)
            raise KeyError(msg)
        else:
            if okey is None:
                raise IndexError("\'{}\' not available.".format(key))

        if columns:
            try:
                odata = okey[columns]
            except KeyError:
                msg = ("At least one of the requested columns does not "
                       "exist.\nAvailable columns: {}").format(okey.columns)
                logger.error(msg)
                raise KeyError(msg)
        else:
            odata = okey

        return(odata)

    def get_dive_details(self, key, columns=None):
        """Accessor for the ``dives`` attribute

        Parameters
        ----------
        key : {"row_ids", "model", "splines", "spline_derivs", crit_vals}
            Name of the key to retrieve.
        columns : array_like, optional
            Names of the columns of the dataframe in `key`, when applicable.

        """
        try:
            okey = self.dives[key]
        except KeyError:
            msg = ("\'{}\' is not found.\nAvailable keys: {}"
                   .format(key, self.dives.keys()))
            logger.error(msg)
            raise KeyError(msg)
        else:
            if okey is None:
                raise IndexError("\'{}\' not available.".format(key))

        if columns:
            try:
                odata = okey[columns]
            except KeyError:
                msg = ("At least one of the requested columns does not "
                       "exist.\nAvailable columns: {}").format(okey.columns)
                logger.error(msg)
                raise KeyError(msg)
        else:
            odata = okey

        return(odata)

    def get_depth(self, kind="measured"):
        """Retrieve depth records

        Parameters
        ----------
        kind : {"measured", "zoc"}
            Which depth to retrieve.

        Returns
        -------
        pandas.Series

        """
        kinds = ["measured", "zoc"]
        if kind == kinds[0]:
            odepth = self.tdr["depth"]
        elif kind == kinds[1]:
            odepth = self.zoc_pars["depth_zoc"]
            if odepth is None:
                msg = "ZOC depth not available."
                logger.error(msg)
                raise IndexError(msg)
        else:
            msg = "kind must be one of: {}".format(kinds)
            logger.error(msg)
            raise IndexError(msg)

        return(odepth.rename("depth"))

    def get_speed(self, kind="measured"):
        """Retrieve speed records

        Parameters
        ----------
        kind : {"measured", "calibrated"}
            Which speed to retrieve.

        Returns
        -------
        pandas.Series

        """
        kinds = ["measured", "calibrated"]
        if kind == kinds[0]:
            ospeed = self.tdr[self.speed_colname]
        elif kind == kinds[1]:
            qfit = self.speed_calib_fit
            if qfit is None:
                msg = "Calibrated speed not available."
                logger.error(msg)
                raise IndexError(msg)
            else:
                coefs = qfit.params
                coef_a = coefs[0]
                coef_b = coefs[1]

            ospeed = (self.tdr[self.speed_colname] - coef_a) / coef_b
        else:
            msg = "kind must be one of: {}".format(kinds)
            logger.error(msg)
            raise IndexError(msg)

        return(ospeed)

    def _get_dive_spline_slot(self, diveNo, name):
        """Accessor for the R objects in `dives`["splines"]

        Private method to retrieve elements easily.  Elements can be
        accessed individually as is, but some elements are handled
        specially.

        Parameters
        ----------
        diveNo : int or float
            Which dive number to retrieve spline details for.
        name : str
            Element to retrieve. {"data", "xy", "knots", "coefficients",
            "order", "lambda.opt", "sigmasq", "degree", "g", "a", "b",
            "variter"}

        """
        # Safe to assume these are all scalars, based on the current
        # default settings in diveMove's `.cutDive`
        scalars = ["order", "lambda.opt", "sigmasq", "degree",
                   "g", "a", "b", "variter"]
        idata = self.get_dive_details("splines")[diveNo]
        if name == "data":
            x = pd.TimedeltaIndex(np.array(idata[name][0]), unit="s")
            odata = pd.Series(np.array(idata[name][1]), index=x)
        elif name == "xy":
            x = pd.TimedeltaIndex(np.array(idata["x"]), unit="s")
            odata = pd.Series(np.array(idata["y"]), index=x)
        elif name in scalars:
            odata = np.float(idata[name][0])
        else:
            odata = np.array(idata[name])

        return(odata)


if __name__ == '__main__':
    tdrX = get_diveMove_sample_data()
    print(tdrX)
