"""TDR objects homologous to `R` package diveMove's main classes

"""

import logging
import numpy as np
import pandas as pd
from skdiveMove.tdrphases import TDRPhases
import skdiveMove.plotting as plotting
import skdiveMove.calibspeed as speedcal
from skdiveMove.helpers import (get_var_sampling_interval,
                                _get_dive_indices, _add_xr_attr,
                                _one_dive_stats, _speed_stats)
import skdiveMove.calibconfig as calibconfig
import xarray as xr


logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())

# Keep attributes in xarray operations
xr.set_options(keep_attrs=True)


class TDR(TDRPhases):
    """Base class encapsulating TDR objects and processing

    TDR subclasses `TDRPhases` to provide comprehensive TDR processing
    capabilities.

    See help(TDR) for inherited attributes.

    Attributes
    ----------
    speed_calib_fit : quantreg model fit
        Model object fit by quantile regression for speed calibration.

    Examples
    --------
    Construct an instance from diveMove example dataset

    >>> from skdiveMove.tests import diveMove2skd
    >>> tdrX = diveMove2skd()

    Plot the `TDR` object

    >>> tdrX.plot()  # doctest: +ELLIPSIS
    (<Figure ... 1 Axes>, <AxesSubplot:...>)

    """

    def __init__(self, *args, **kwargs):
        """Set up attributes for TDR objects

        Parameters
        ----------
        *args : positional arguments
            Passed to :meth:`TDRSource.__init__`
        **kwargs : keyword arguments
            Passed to :meth:`TDRSource.__init__`

        """
        TDRPhases.__init__(self, *args, **kwargs)

        # Speed calibration fit
        self.speed_calib_fit = None

    def __str__(self):
        base = TDRPhases.__str__(self)

        speed_fmt_pref = "Speed calibration coefficients:"
        if self.speed_calib_fit is not None:
            speed_ccoef_a, speed_ccoef_b = self.speed_calib_fit.params
            speed_coefs_fmt = ("\n{0:<20} (a={1:.4f}, b={2:.4f})"
                               .format(speed_fmt_pref,
                                       speed_ccoef_a, speed_ccoef_b))
        else:
            speed_ccoef_a, speed_ccoef_b = (None, None)
            speed_coefs_fmt = ("\n{0:<20} (a=None, b=None)"
                               .format(speed_fmt_pref))

        return(base + speed_coefs_fmt)

    def calibrate_speed(self, tau=0.1, contour_level=0.1, z=0, bad=[0, 0],
                        **kwargs):
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
        **kwargs : optional keyword arguments
            Passed to :func:`~speedcal.calibrate_speed`

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.calibrate_speed(z=2)

        """
        depth = self.get_depth("zoc").to_series()
        ddiffs = depth.reset_index().diff().set_index(depth.index)
        ddepth = ddiffs["depth"].abs()
        rddepth = ddepth / ddiffs["date_time"].dt.total_seconds()
        curspeed = self.get_speed("measured").to_series()
        ok = (ddepth > z) & (rddepth > bad[0]) & (curspeed > bad[1])
        rddepth = rddepth[ok]
        curspeed = curspeed[ok]

        kde_data = pd.concat((rddepth.rename("depth_rate"),
                              curspeed), axis=1)
        qfit, ax = speedcal.calibrate_speed(kde_data, tau=tau,
                                            contour_level=contour_level,
                                            z=z, bad=bad, **kwargs)
        self.speed_calib_fit = qfit
        logger.info("Finished calibrating speed")

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

        Examples
        --------
        ZOC using the "filter" method

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()

        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.dive_stats()  # doctest: +ELLIPSIS
                        begdesc  ... postdive_mean_speed
        1   2002-01-05 ...                      1.398859
        2   ...

        """
        phases_df = self.get_dives_details("row_ids")

        # calib_speed=False if no fit object
        tdr = self.get_tdr(calib_depth=True,
                           calib_speed=bool(self.speed_calib_fit))

        intvl = (get_var_sampling_interval(tdr[self.depth_name])
                 .total_seconds())
        tdr = tdr.to_dataframe()

        dive_ids = phases_df.loc[:, "dive_id"]
        postdive_ids = phases_df.loc[:, "postdive_id"]
        ok = (dive_ids > 0) & dive_ids.isin(postdive_ids)
        okpd = (postdive_ids > 0) & postdive_ids.isin(dive_ids)
        postdive_ids = postdive_ids[okpd]

        postdive_dur = (postdive_ids.reset_index()
                        .groupby("postdive_id")
                        .apply(lambda x: x.iloc[-1] - x.iloc[0]))

        tdrf = (pd.concat((phases_df[["dive_id", "dive_phase"]][ok],
                           tdr[ok]), axis=1).reset_index())

        # Ugly hack to re-order columns for `diveMove` convention
        names0 = ["dive_id", "dive_phase", "date_time", self.depth_name]
        colnames = tdrf.columns.to_list()
        if self.has_speed:
            names0.append(self.speed_name)
        colnames = names0 + list(set(colnames) - set(names0))
        tdrf = tdrf.reindex(columns=colnames)
        tdrf_grp = tdrf.groupby("dive_id")

        ones_list = []
        for name, grp in tdrf_grp:
            res = _one_dive_stats(grp, interval=intvl,
                                  has_speed=self.has_speed)
            # Rename to match dive number
            res = res.rename({0: name})

            if depth_deriv:
                deriv_stats = self._get_dive_deriv_stats(name)
                res = pd.concat((res, deriv_stats), axis=1)

            ones_list.append(res)

        ones_df = pd.concat(ones_list, ignore_index=True)
        ones_df.set_index(dive_ids[ok].unique(), inplace=True)
        ones_df.index.rename("dive_id", inplace=True)
        ones_df["postdive_dur"] = postdive_dur["date_time"]

        # For postdive total distance and mean speed (if available)
        if self.has_speed:
            speed_postd = (tdr[self.speed_name][okpd]
                           .groupby(postdive_ids))
            pd_speed_ll = []
            for name, grp in speed_postd:
                res = _speed_stats(grp.reset_index())
                onames = ["postdive_tdist", "postdive_mean_speed"]
                res_df = pd.DataFrame(res[:, :-1], columns=onames,
                                      index=[name])
                pd_speed_ll.append(res_df)
            pd_speed_stats = pd.concat(pd_speed_ll)
            ones_df = pd.concat((ones_df, pd_speed_stats), axis=1)

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
            Arguments passed to plotting function.

        Returns
        -------
        tuple
            :class:`~matplotlib.figure.Figure`,
            :class:`~matplotlib.axes.Axes` instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
        ...           depth_lim=[95, -1])  # doctest: +ELLIPSIS
        (<Figure ... with 1 Axes>, <AxesSubplot:...'>)

        """
        try:
            depth = self.get_depth("zoc")
        except LookupError:
            depth = self.get_depth("measured")

        if "ylab_depth" not in kwargs:
            ylab_depth = ("{0} [{1}]"
                          .format(depth.attrs["full_name"],
                                  depth.attrs["units"]))
            kwargs.update(ylab_depth=ylab_depth)

        depth = depth.to_series()

        if concur_vars is None:
            fig, ax = plotting.plot_tdr(depth, **kwargs)
        elif concur_var_titles is None:
            ccvars = self.tdr[concur_vars].to_dataframe()
            fig, ax = plotting.plot_tdr(depth, concur_vars=ccvars, **kwargs)
        else:
            ccvars = self.tdr[concur_vars].to_dataframe()
            ccvars_title = concur_var_titles  # just to shorten
            fig, ax = plotting.plot_tdr(depth,
                                        concur_vars=ccvars,
                                        concur_var_titles=ccvars_title,
                                        **kwargs)

        return(fig, ax)

    def plot_zoc(self, xlim=None, ylim=None, **kwargs):
        """Plot zero offset correction filters

        Parameters
        ----------
        xlim, ylim : 2-tuple/list, optional
            Minimum and maximum limits for ``x``- and ``y``-axis,
            respectively.
        **kwargs : optional keyword arguments
            Passed to :func:`~matplotlib.pyplot.subplots`.

        Returns
        -------
        tuple
            :class:`~matplotlib.figure.Figure`,
            :class:`~matplotlib.axes.Axes` instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.zoc("filter", k=K, probs=P, depth_bounds=DB)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.plot_zoc()  # doctest: +ELLIPSIS
        (<Figure ... with 3 Axes>, array([<AxesSubplot:...'>,
            <AxesSubplot:...'>, <AxesSubplot:...>], dtype=object))

        """
        zoc_method = self.zoc_method
        depth_msrd = self.get_depth("measured")
        ylab = ("{0} [{1}]"
                .format(depth_msrd.attrs["full_name"],
                        depth_msrd.attrs["units"]))

        if zoc_method == "filter":
            zoc_filters = self.zoc_filters
            depth = depth_msrd.to_series()
            if "ylab" not in kwargs:
                kwargs.update(ylab=ylab)

            fig, ax = (plotting
                       ._plot_zoc_filters(depth, zoc_filters, xlim, ylim,
                                          **kwargs))
        elif zoc_method == "offset":
            depth_msrd = depth_msrd.to_series()
            depth_zoc = self.get_depth("zoc").to_series()
            fig, ax = plotting.plt.subplots(1, 1, **kwargs)
            ax = depth_msrd.plot(ax=ax, rot=0, label="measured")
            depth_zoc.plot(ax=ax, label="zoc")
            ax.axhline(0, linestyle="--", linewidth=0.75, color="k")
            ax.set_xlabel("")
            ax.set_ylabel(ylab)
            ax.legend(loc="lower right")
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.invert_yaxis()

        return(fig, ax)

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
            Arguments passed to plotting function.

        Returns
        -------
        tuple
            :class:`~matplotlib.figure.Figure`,
            :class:`~matplotlib.axes.Axes` instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.plot_phases(list(range(250, 300)),
        ...                  surface=True)  # doctest: +ELLIPSIS
        (<Figure ... with 1 Axes>, <AxesSubplot:...>)

        """
        row_ids = self.get_dives_details("row_ids")
        dive_ids = row_ids["dive_id"]
        dive_ids_uniq = dive_ids.unique()
        postdive_ids = row_ids["postdive_id"]

        if diveNo is None:
            diveNo = np.arange(1, row_ids["dive_id"].max() + 1).tolist()
        else:
            diveNo = [x for x in sorted(diveNo) if x in dive_ids_uniq]

        depth_all = self.get_depth("zoc").to_dataframe()  # DataFrame

        if concur_vars is None:
            dives_all = depth_all
        else:
            concur_df = self.tdr.to_dataframe().loc[:, concur_vars]
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

        wet_dry = self.time_budget(ignore_z=True, ignore_du=True)
        drys = wet_dry[wet_dry["phase_label"] == "L"][["beg", "end"]]
        if (drys.shape[0] > 0):
            dry_time = drys
        else:
            dry_time = None

        if concur_vars is None:
            fig, ax = (plotting
                       .plot_tdr(dives_df.iloc[:, 0],
                                 phase_cat=details_df["dive_phase"],
                                 dry_time=dry_time, **kwargs))
        else:
            fig, ax = (plotting
                       .plot_tdr(dives_df.iloc[:, 0],
                                 concur_vars=dives_df.iloc[:, 1:],
                                 concur_var_titles=concur_var_titles,
                                 phase_cat=details_df["dive_phase"],
                                 dry_time=dry_time, **kwargs))

        return(fig, ax)

    def plot_dive_model(self, diveNo=None, **kwargs):
        """Plot dive model for selected dive

        Parameters
        ----------
        diveNo : array_like, optional
            List of dive numbers (1-based) to plot.
        **kwargs : optional keyword arguments
            Arguments passed to plotting function.

        Returns
        -------
        tuple
            :class:`~matplotlib.figure.Figure`,
            :class:`~matplotlib.axes.Axes` instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.plot_dive_model(diveNo=20,
        ...                      figsize=(10, 10))  # doctest: +ELLIPSIS
        (<Figure ... with 2 Axes>, (<AxesSubplot:...>, <AxesSubplot:...>))

        """
        dive_ids = self.get_dives_details("row_ids", "dive_id")
        crit_vals = self.get_dives_details("crit_vals").loc[diveNo]
        idxs = _get_dive_indices(dive_ids, diveNo)
        depth = self.get_depth("zoc").to_dataframe().iloc[idxs]
        depth_s = self._get_dive_spline_slot(diveNo, "xy")
        depth_deriv = (self.get_dives_details("spline_derivs").loc[diveNo])

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
        d_crit = crit_vals["descent_crit"].astype(int)
        a_crit = crit_vals["ascent_crit"].astype(int)
        d_crit_rate = crit_vals["descent_crit_rate"]
        a_crit_rate = crit_vals["ascent_crit_rate"]
        title = "Dive: {:d}".format(diveNo)
        fig, axs = plotting.plot_dive_model(depth, depth_s=depth_s,
                                            depth_deriv=depth_deriv,
                                            d_crit=d_crit, a_crit=a_crit,
                                            d_crit_rate=d_crit_rate,
                                            a_crit_rate=a_crit_rate,
                                            leg_title=title, **kwargs)

        return(fig, axs)

    def get_depth(self, kind="measured"):
        """Retrieve depth records

        Parameters
        ----------
        kind : {"measured", "zoc"}
            Which depth to retrieve.

        Returns
        -------
        xarray.DataArray

        """
        kinds = ["measured", "zoc"]
        if kind == kinds[0]:
            odepth = self.depth
        elif kind == kinds[1]:
            odepth = self.depth_zoc
            if odepth is None:
                msg = "ZOC depth not available."
                logger.error(msg)
                raise LookupError(msg)
        else:
            msg = "kind must be one of: {}".format(kinds)
            logger.error(msg)
            raise LookupError(msg)

        return(odepth)

    def get_speed(self, kind="measured"):
        """Retrieve speed records

        Parameters
        ----------
        kind : {"measured", "calibrated"}
            Which speed to retrieve.

        Returns
        -------
        xarray.DataArray

        """
        kinds = ["measured", "calibrated"]
        ispeed = self.speed

        if kind == kinds[0]:
            ospeed = ispeed
        elif kind == kinds[1]:
            qfit = self.speed_calib_fit
            if qfit is None:
                msg = "Calibrated speed not available."
                logger.error(msg)
                raise LookupError(msg)
            else:
                coefs = qfit.params
                coef_a = coefs[0]
                coef_b = coefs[1]

            ospeed = (ispeed - coef_a) / coef_b
            _add_xr_attr(ospeed, "history", "speed_calib_fit")
        else:
            msg = "kind must be one of: {}".format(kinds)
            logger.error(msg)
            raise LookupError(msg)

        return(ospeed)

    def get_tdr(self, calib_depth=True, calib_speed=True):
        """Return a copy of tdr Dataset

        Parameters
        ----------
        calib_depth : bool, optional
            Whether to return calibrated depth measurements.
        calib_speed : bool, optional
            Whether to return calibrated speed measurements.

        Returns
        -------
        xarray.Dataset

        """
        tdr = self.tdr.copy()

        if calib_depth:
            depth_name = self.depth_name
            depth_cal = self.get_depth("zoc")
            tdr[depth_name] = depth_cal

        if self.has_speed and calib_speed:
            speed_name = self.speed_name
            speed_cal = self.get_speed("calibrated")
            tdr[speed_name] = speed_cal

        return(tdr)

    def extract_dives(self, diveNo, **kwargs):
        """Extract TDR data corresponding to a particular set of dives

        Parameters
        ----------
        diveNo : array_like, optional
            List of dive numbers (1-based) to plot.
        **kwargs : optional keyword arguments
            Passed to :meth:`get_tdr`

        Returns
        -------
        xarray.Dataset

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd(has_speed=False)
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.extract_dives(diveNo=20)  # doctest: +ELLIPSIS
        <xarray.Dataset>
        Dimensions: ...

        """
        dive_ids = self.get_dives_details("row_ids", "dive_id")
        idxs = _get_dive_indices(dive_ids, diveNo)
        tdr = self.get_tdr(**kwargs)
        tdr_i = tdr[dict(date_time=idxs.astype(int))]

        return(tdr_i)


def calibrate(tdr_file, config_file=None):
    """Perform all major TDR calibration operations

    Detect periods of major activities in a `TDR` object, calibrate depth
    readings, and speed if appropriate, in preparation for subsequent
    summaries of diving behaviour.

    This function is a convenience wrapper around :meth:`TDR.detect_wet`,
    :meth:`TDR.detect_dives`, :meth:`TDR.detect_dive_phases`,
    :meth:`TDR.zoc`, and :meth:`TDR.calibrate_speed`.  It performs wet/dry
    phase detection, zero-offset correction of depth, detection of dives,
    as well as proper labelling of the latter, and calibrates speed data if
    appropriate.

    Due to the complexity of this procedure, and the number of settings
    required for it, a calibration configuration file (JSON) is used to
    guide the operations.

    Parameters
    ----------
    tdr_file : str, Path or xarray.backends.*DataStore
        As first argument for :func:`xarray.load_dataset`.
    config_file : str
        A valid string path for TDR calibration configuration file.

    Returns
    -------
    out : TDR

    See Also
    --------
    dump_config_template : configuration template

    """
    if config_file is None:
        config = calibconfig._DEFAULT_CONFIG
    else:
        config = calibconfig.read_config(config_file)

    logger = logging.getLogger(__name__)
    logger.setLevel(config["log_level"])

    load_dataset_kwargs = config["read"].pop("load_dataset_kwargs")
    logger.info("Reading config: {}, {}"
                .format(config["read"], load_dataset_kwargs))
    tdr = TDR(tdr_file, **config["read"], **load_dataset_kwargs)

    do_zoc = config["zoc"].pop("required")
    if do_zoc:
        logger.info("ZOC config: {}".format(config["zoc"]))
        tdr.zoc(config["zoc"]["method"], **config["zoc"]["parameters"])

    logger.info("Wet/Dry config: {}".format(config["wet_dry"]))
    tdr.detect_wet(**config["wet_dry"])

    logger.info("Dives config: {}".format(config["dives"]))
    tdr.detect_dives(config["dives"].pop("dive_thr"))
    tdr.detect_dive_phases(**config["dives"])

    do_speed_calib = bool(config["speed_calib"].pop("required"))
    if do_speed_calib:
        logger.info("Speed calibration config: {}"
                    .format(config["speed_calib"]))
        tdr.calibrate_speed(**config["speed_calib"], plot=False)

    return(tdr)


if __name__ == '__main__':
    # Set up info level logging
    logging.basicConfig(level=logging.INFO)

    ifile = r"tests/data/ag_mk7_2002_022.nc"
    tdrX = TDR(ifile, has_speed=True)
    # tdrX = TDRSource(ifile, has_speed=True)
    # print(tdrX)
