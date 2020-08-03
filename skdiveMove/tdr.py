"""TDR objects homologous to `R` package diveMove's main classes

The :class:`TDR` class aims to be a comprehensive class to encapsulate the
processing of `TDR` records from a data file.

This module instantiates an `R` session to interact with low-level
functions and methods of package `diveMove`.

Class & Main Methods Summary
----------------------------

See `API` section for details on minor methods.

Calibration and phase detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autosummary::

   TDR
   TDR.zoc
   TDR.detect_wet
   TDR.detect_dives
   TDR.detect_dive_phases
   TDR.calibrate
   TDR.calibrate_speed

Analyses
~~~~~~~~

.. autosummary::

   TDR.dive_stats
   TDR.time_budget
   TDR.stamp_dives

Plotting
~~~~~~~~

.. autosummary::

   TDR.plot
   TDR.plot_zoc
   TDR.plot_phases
   TDR.plot_dive_model

API
---

"""

import logging
import numpy as np
import pandas as pd
from skdiveMove.tdrsource import TDRSource
from skdiveMove.zoc import ZOC
from skdiveMove.tdrphases import TDRPhases
import skdiveMove.plotting as plotting
import skdiveMove.calibrate_speed as speedcal
from skdiveMove.helpers import (get_var_sampling_interval,
                                _get_dive_indices, _add_xr_attr,
                                _one_dive_stats, _speed_stats)
import xarray as xr


logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())

# Keep attributes in xarray operations
xr.set_options(keep_attrs=True)


class TDR(TDRSource):
    """Base class encapsulating TDR objects and processing

    TDR subclasses `TDRSource` to provide comprehensive TDR processing
    capabilities.

    Attributes
    ----------
    tdr_file : str
        String indicating the file where data comes from.
    tdr : xarray.Dataset
        Dataset with input data.
    depth_name : str
        Name of Dataset variable with depth measurements.
    has_speed : bool
        Whether input data include speed measurements.
    speed_name : str
        Name of Dataset variable with the speed measurements.
    zoc_depth : ZOC
        Instance to perform and store zero offset correction operations for
        depth.
    phases : TDRPhases
        Instance for performing wet/dry and dive phase detection.
    speed_calib_fit : quantreg model fit
        Model object fit by quantile regression for speed calibration.

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

    def __init__(self, *args, **kwargs):
        """Set up attributes for TDR objects

        """
        TDRSource.__init__(self, *args, **kwargs)

        # Attributes to be set from other methods: method used, corrected
        # depth, filtered depth when method="filter"
        self.zoc_depth = ZOC()
        # Speed calibration fit
        self.speed_calib_fit = None
        # Wet phase activity identification
        self.phases = TDRPhases()

    def __str__(self):
        x = self.tdr
        xdf = x.to_dataframe()
        objcls = ("Time-Depth Recorder data -- Class {} object\n"
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
        zocm = "{0:<20}: \'{1}\'\n".format("ZOC method",
                                           self.zoc_depth.method)
        attr_list = "Attributes:\n"
        for key, val in sorted(x.attrs.items()):
            attr_list += "{0:>35}: {1}\n".format(key, val)
        attr_list = attr_list.rstrip("\n")

        return(objcls + src + itv + nsamples + beg + end + dur + drange +
               others + zocm + attr_list)

    def zoc(self, method="filter", **kwargs):
        """Zero offset correction

        Set the ``zoc_depth`` attribute.

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

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)

        Using the "filter" method

        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.zoc(k=K, probs=P, depth_bounds=DB)

        Plot the filters that were applied

        >>> tdrX.plot_zoc(ylim=[-1, 10])

        """
        depth = self.get_depth("measured")
        self.zoc_depth(depth, method=method, **kwargs)
        logger.info("Finished ZOC")

    def detect_wet(self, interp_wet=False, **kwargs):
        """Detect wet/dry activity phases

        Parameters
        ----------
        interp_wet : bool, optional
        **kwargs : Keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.detect_wet`

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.  Unlike
        `diveMove`, the beginning/ending times for each phase are not
        stored with the class instance, as this information can be
        retrieved via the `.time_budget` method.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases

        >>> tdrX.detect_wet()

        Access the "phases" and "dry_thr" attributes

        >>> tdrX.get_wet_activity("phases")
        >>> tdrX.get_wet_activity("dry_thr")

        """
        depth = self.get_depth("zoc")
        self.phases.detect_wet(depth, **kwargs)

        if interp_wet:
            zdepth = depth.to_series()
            phases = self.phases.wet_dry
            iswet = phases["phase_label"] == "W"
            iswetna = iswet & zdepth.isna()

            if any(iswetna):
                depth_intp = zdepth[iswet].interpolate(method="cubic")
                zdepth[iswetna] = np.maximum(np.zeros_like(depth_intp),
                                             depth_intp)
                zdepth = zdepth.to_xarray()
                zdepth.attrs = depth.attrs
                _add_xr_attr(zdepth, "history", "interp_wet")
                self.zoc_depth._depth_zoc = zdepth
                self.zoc_depth._params.update(dict(interp_wet=interp_wet))

        logger.info("Finished detecting wet/dry periods")

    def detect_dives(self, dive_thr):
        """Identify dive events

        Set the ``dives`` attribute's "row_ids" dictionary element, and
        update the ``wet_act`` attribute's "phases" dictionary element.

        Parameters
        ----------
        dry_thr : float
            Passed to :meth:`~tdrphases.TDRPhases.detect_dives`.

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        """
        depth = self.get_depth("zoc")
        self.phases.detect_dives(depth, dive_thr=dive_thr)
        logger.info("Finished detecting dives")

    def detect_dive_phases(self, **kwargs):
        """Detect dive phases

        Complete filling the ``dives`` attribute.

        Parameters
        ----------
        **kwargs : optional keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.detect_dive_phases`

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        Detect dive phases using the "unimodal" method and selected
        parameters

        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)

        """
        depth = self.get_depth("zoc")
        self.phases.detect_dive_phases(depth, **kwargs)
        logger.info("Finished detecting dive phases")

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
        **kwargs : optional keyword arguments
            Passed to :meth:`TDR.zoc` method:

            * method 'filter': ('k', 'probs', 'depth_bounds' (defaults to
              range), 'na_rm' (defaults to True)).
            * method 'offset': ('offset').

        Notes
        -----
        This method is homologous to diveMove's ``calibrateDepth`` function.

        Examples
        --------
        ZOC using the "filter" method

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()

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
        self.detect_dives(dive_thr=dive_thr)
        self.detect_dive_phases(dive_model=dive_model,
                                smooth_par=smooth_par,
                                knot_factor=knot_factor,
                                descent_crit_q=descent_crit_q,
                                ascent_crit_q=ascent_crit_q)

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
            Passed to :func:`calibrate_speed.calibrate`

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
        qfit, ax = speedcal.calibrate(kde_data, tau=tau,
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
        >>> tdrX.calibrate(dive_thr=3, zoc_method="filter",
        ...                k=K, probs=P, depth_bounds=DB,
        ...                descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.dive_stats()

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
                deriv_stats = self.phases._get_dive_deriv_stats(name)
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
            ``matplotlib.pyplot`` Figure and Axes instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.plot(xlim=["2002-01-05 21:00:00", "2002-01-06 04:10:00"],
        ...           depth_lim=[95, -1])

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
            Passed to `matplotlib.pyplot.subplots`.

        Returns
        -------
        tuple
            `matplotlib.pyplot` Figure and Axes instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.plot_zoc()

        """
        zoc_method = self.zoc_depth.method
        depth_msrd = self.get_depth("measured")
        ylab = ("{0} [{1}]"
                .format(depth_msrd.attrs["full_name"],
                        depth_msrd.attrs["units"]))

        if zoc_method == "filter":
            zoc_filters = self.zoc_depth.filters
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
            `matplotlib.pyplot` Figure and Axes instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.calibrate(dive_thr=3, zoc_method="offset",
        ...                offset=3, descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.plot_phases(list(range(250, 300)), surface=True)

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
            Pyplot Figure and Axes instances.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.calibrate(dive_thr=3, zoc_method="offset",
        ...                offset=3, descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.plot_dive_model(diveNo=20, figsize=(10, 10))

        """
        dive_ids = self.get_dives_details("row_ids", "dive_id")
        crit_vals = self.get_dives_details("crit_vals").loc[diveNo]
        idxs = _get_dive_indices(dive_ids, diveNo)
        depth = self.get_depth("zoc").to_dataframe().iloc[idxs]
        depth_s = self.phases._get_dive_spline_slot(diveNo, "xy")
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
            odepth = self.zoc_depth.depth
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

    def _get_wet_activity(self):
        return(self.phases.wet_dry)

    wet_dry = property(_get_wet_activity)
    """Wet/dry activity labels

    Extends :meth:`TDRPhases.wet_dry`.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: `phase_id` and `phase_label` for each
        measurement.

    """

    def get_dives_details(self, *args, **kwargs):
        """Retrieve wet/dry activity DataFrame

        Parameters
        ----------
        *args : positional arguments
            Passed to :meth:`~tdrphases.TDRPhases.get_dives_details`
        **kwargs : keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.get_dives_details`

        """
        return(self.phases.get_dives_details(*args, **kwargs))

    def get_dive_deriv(self, *args, **kwargs):
        """Retrieve depth spline derivative for a given dive

        Parameters
        ----------
        *args : arguments
            Passed to :meth:`~tdrphases.TDRPhases.get_dive_deriv`
        **kwargs : keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.get_dive_deriv`

        """
        return(self.phases.get_dive_deriv(*args, **kwargs))

    def get_phases_params(self, key):
        """Retrieve parameters used for identification of phases

        Parameters
        ----------
        key : {'wet_dry', 'dives'}
            Name of type of parameters to retrieve.

        Returns
        -------
        out : dict

        """
        return(self.phases.get_params(key))

    def _get_zoc_params(self):
        return(self.zoc_depth.params)

    zoc_params = property(_get_zoc_params)
    """ZOC procedure parameters

    Extends :attr:`ZOC.params`.

    Returns
    -------
    method : str
        Method used for ZOC.
    params : dict
        Dictionary with parameters and values used for ZOC.

    """

    def time_budget(self, **kwargs):
        """Summary of wet/dry activities at the broadest scale

        Parameters
        ----------
        **kwargs : optional keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.time_budget`

        Returns
        -------
        out : pandas.DataFrame

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.calibrate(dive_thr=3, zoc_method="offset",
        ...                offset=3, descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.time_budget(ignore_z=True, ignore_du=True)

        """
        return(self.phases.time_budget(**kwargs))

    def stamp_dives(self, **kwargs):
        """Identify the wet/dry activity phase corresponding to each dive

        Parameters
        ----------
        **kwargs : optional keyword arguments
            Passed to :meth:`~tdrphases.TDRPhases.stamp_dives`

        Returns
        -------
        out : pandas.DataFrame

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.calibrate(dive_thr=3, zoc_method="offset",
        ...                offset=3, descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.stamp_dives(ignore_z=True)

        """
        return(self.phases.stamp_dives(**kwargs))

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
        >>> tdrX = diveMove2skd()
        >>> tdrX.calibrate(dive_thr=3, zoc_method="offset",
        ...                offset=3, descent_crit_q=0.01, knot_factor=20)
        >>> tdrX.extract_dive(diveNo=20)

        """
        dive_ids = self.get_dives_details("row_ids", "dive_id")
        idxs = _get_dive_indices(dive_ids, diveNo)
        tdr = self.get_tdr(**kwargs)
        tdr_i = tdr[dict(date_time=idxs.astype(int))]

        return(tdr_i)


if __name__ == '__main__':
    # Set up info level logging
    logging.basicConfig(level=logging.INFO)

    ifile = r"tests/data/ag_mk7_2002_022.nc"
    tdrX = TDR(ifile, has_speed=True)
    # tdrX = TDRSource(ifile, has_speed=True)
    # print(tdrX)
