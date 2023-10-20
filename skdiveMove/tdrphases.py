"""Class handling all TDR phase operations

Phase identification methods take `depth` as input.

Class and Methods Summary
-------------------------

.. autosummary::

   TDRPhases.detect_wet
   TDRPhases.detect_dives
   TDRPhases.detect_dive_phases
   TDRPhases.get_dives_details
   TDRPhases.get_dive_deriv
   TDRPhases.wet_dry
   TDRPhases.get_phases_params
   TDRPhases.time_budget
   TDRPhases.stamp_dives

"""

import logging
import numpy as np
import pandas as pd
from skdiveMove.zoc import ZOC
from skdiveMove.core import diveMove, robjs, pandas2ri
from skdiveMove.helpers import (get_var_sampling_interval, _cut_dive,
                                rle_key, _append_xr_attr)

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


class TDRPhases(ZOC):
    """Core TDR phase identification routines

    See help(TDRSource) for inherited attributes.

    Attributes
    ----------
    wet_dry

    dives : dict
        Dictionary of dive activity data {'row_ids': pandas.DataFrame,
        'model': str, 'splines': dict, 'spline_derivs': pandas.DataFrame,
        'crit_vals': pandas.DataFrame}.
    params : dict
        Dictionary with parameters used for detection of wet/dry and dive
        phases. {'wet_dry': {'dry_thr': float, 'wet_thr': float}, 'dives':
        {'dive_thr': float, 'dive_model': str, 'smooth_par': float,
        'knot_factor': int, 'descent_crit_q': float, 'ascent_crit_q':
        float}}

    """
    def __init__(self, *args, **kwargs):
        """Initialize TDRPhases instance

        Parameters
        ----------
        *args : positional arguments
            Passed to :meth:`ZOC.__init__`
        **kwargs : keyword arguments
            Passed to :meth:`ZOC.__init__`

        """
        ZOC.__init__(self, *args, **kwargs)
        self._wet_dry = None
        self.dives = dict(row_ids=None, model=None, splines=None,
                          spline_derivs=None, crit_vals=None)
        self.params = dict(wet_dry={}, dives={})

    def __str__(self):
        base = ZOC.__str__(self)
        wetdry_params = self.get_phases_params("wet_dry")
        dives_params = self.get_phases_params("dives")
        return (base +
                ("\n{0:<20} {1}\n{2:<20} {3}"
                 .format("Wet/Dry parameters:", wetdry_params,
                         "Dives parameters:", dives_params)))

    def detect_wet(self, dry_thr=70, wet_cond=None, wet_thr=3610,
                   interp_wet=False):
        """Detect wet/dry activity phases

        A categorical variable is then created with value ``L`` (dry) for
        rows with null depth samples and value ``W`` (wet) otherwise. This
        assumes that TDRs were programmed to turn off recording of depth
        when instrument is dry (typically by means of a salt-water
        switch). If this assumption cannot be made for any reason, then a
        boolean vector as long as the time series can be supplied to
        indicate which observations should be considered wet. The duration
        of each of these phases of activity is subsequently calculated.  If
        the duration of a dry phase (``L``) is less than a threshold
        (configuration variable `dry_thr`), then the values in the factor
        for that phase are changed to ``W`` (wet). The duration of phases
        is then recalculated, and if the duration of a phase of wet
        activity is less than another threshold (variable `wet_thr`), then
        the corresponding value for the factor is changed to ``Z`` (trivial
        wet). The durations of all phases are recalculated a third time to
        provide final phase durations.

        Some instruments produce a peculiar pattern of missing data near
        the surface, at the beginning and/or end of dives. The argument
        ``interp_wet`` may help to rectify this problem by using an
        interpolating spline function to impute the missing data,
        constraining the result to a minimum depth of zero.  Please note
        that this optional step is performed after ZOC and before
        identifying dives, so that interpolation is performed through dry
        phases coded as wet because their duration was briefer than
        `dry_thr`.  Therefore, `dry_thr` must be chosen carefully to avoid
        interpolation through legitimate dry periods.

        Parameters
        ----------
        dry_thr : float, optional
            Dry error threshold in seconds. Dry phases shorter than this
            threshold will be considered as wet.
        wet_cond : bool mask, optional
            A Pandas.Series bool mask indexed as `depth`. It indicates
            which observations should be considered wet. If it is not
            provided, records with non-missing depth are assumed to
            correspond to wet conditions. Default is generated from testing
            for non-missing `depth`.
        wet_thr : float, optional
            Wet threshold in seconds. At-sea phases shorter than this
            threshold will be considered as trivial wet.
        interp_wet : bool, optional
            If `True`, then an interpolating spline function is used to
            impute NA depths in wet periods (after ZOC). Use with
            caution: it may only be useful in cases where the missing data
            pattern in wet periods is restricted to shallow depths near the
            beginning and end of dives. This pattern is common in some
            satellite-linked `TDRs`.

        Notes
        -----

        Unlike `diveMove`, the beginning/ending times for each phase are
        not stored with the class instance, as this information can be
        retrieved via the :meth:`~TDR.time_budget` method.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd("TDRPhases")
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases

        >>> tdrX.detect_wet()

        Access the "phases" and "dry_thr" attributes

        >>> tdrX.wet_dry  # doctest: +ELLIPSIS
                             phase_id phase_label
        date_time
        2002-01-05 ...              1           L
        ...

        """
        # Retrieve copy of depth from our own property
        depth = self.depth_zoc
        depth_py = depth.to_series()
        time_py = depth_py.index
        dtime = get_var_sampling_interval(depth).total_seconds()

        if wet_cond is None:
            wet_cond = ~depth_py.isna()

        phases_l = (diveMove
                    ._detPhase(robjs.vectors.POSIXct(time_py),
                               robjs.vectors.FloatVector(depth_py),
                               dry_thr=dry_thr,
                               wet_thr=wet_thr,
                               wet_cond=(robjs.vectors
                                         .BoolVector(~depth_py.isna())),
                               interval=dtime))
        with (robjs.default_converter + pandas2ri.converter).context():
            phases = pd.DataFrame({'phase_id': phases_l.rx2("phase.id"),
                                   'phase_label': phases_l.rx2("activity")},
                                  index=time_py)

        phases["phase_id"] = phases["phase_id"].astype(int)
        self._wet_dry = phases
        wet_dry_params = dict(dry_thr=dry_thr, wet_thr=wet_thr)
        self.params["wet_dry"].update(wet_dry_params)

        if interp_wet:
            zdepth = depth.to_series()
            iswet = phases["phase_label"] == "W"
            iswetna = iswet & zdepth.isna()

            if any(iswetna):
                depth_intp = zdepth[iswet].interpolate(method="cubic")
                zdepth[iswetna] = np.maximum(np.zeros_like(depth_intp),
                                             depth_intp)
                zdepth = zdepth.to_xarray()
                zdepth.attrs = depth.attrs
                _append_xr_attr(zdepth, "history", "interp_wet")
                self._depth_zoc = zdepth
                self._zoc_params.update(dict(interp_wet=interp_wet))

        logger.info("Finished detecting wet/dry periods")

    def detect_dives(self, dive_thr):
        """Identify dive events

        Set the ``dives`` attribute's "row_ids" dictionary element, and
        update the ``wet_act`` attribute's "phases" dictionary
        element. Whenever the zero-offset corrected depth in an underwater
        phase is below the specified dive threshold.  A new categorical
        variable with finer levels of activity is thus generated, including
        ``U`` (underwater), and ``D`` (diving) in addition to the ones
        described above.

        Once dives have been detected and assigned to a period of wet
        activity, phases within dives are identified using the descent,
        ascent and wiggle criteria (see Detection of dive phases below).
        This procedure generates a categorical variable with levels ``D``,
        ``DB``, ``B``, ``BA``, ``DA``, ``A``, and ``X``, breaking the input
        into descent, descent/bottom, bottom, bottom/ascent, ascent,
        descent/ascent (occurring when no bottom phase can be detected) and
        non-dive (surface), respectively.

        Parameters
        ----------
        dive_thr : float
            Threshold depth below which an underwater phase should be
            considered a dive.

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd("TDRPhases")
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        """
        # Retrieve copy of depth from our own property
        depth = self.depth_zoc
        depth_py = depth.to_series()
        act_phases = self.wet_dry["phase_label"]
        with (robjs.default_converter + pandas2ri.converter).context():
            phases_df = diveMove._detDive(pd.Series(depth_py),
                                          pd.Series(act_phases),
                                          dive_thr=dive_thr)

        # Replace dots with underscore
        phases_df.columns = (phases_df.columns.str
                             .replace(".", "_", regex=False))
        phases_df.set_index(depth_py.index, inplace=True)
        dive_activity = phases_df.pop("dive_activity")
        # Dive and post-dive ID should be integer
        phases_df = phases_df.astype(int)
        self.dives["row_ids"] = phases_df
        self._wet_dry["phase_label"] = dive_activity
        self.params["dives"].update({'dive_thr': dive_thr})

        logger.info("Finished detecting dives")

    def detect_dive_phases(self, dive_model, smooth_par=0.1,
                           knot_factor=3, descent_crit_q=0,
                           ascent_crit_q=0):
        r"""Detect dive phases

        Complete filling the `dives` attribute. The process for each dive
        begins by taking all observations below the dive detection
        threshold, and setting the beginning and end depths to zero, at
        time steps prior to the first and after the last, respectively.
        The latter ensures that descent and ascent derivatives are
        non-negative and non-positive, respectively, so that the end and
        beginning of these phases are not truncated. The next step is to
        fit a model to each dive. Two models can be chosen for this
        purpose: `unimodal` (default) and `smooth.spline` (see Notes).

        Both models consist of a cubic spline, and its first derivative is
        evaluated to investigate changes in vertical rate. Therefore, at
        least 4 observations are required for each dive, so the time series
        is linearly interpolated at equally spaced time steps if this limit
        is not achieved in the current dive. Wiggles at the beginning and
        end of the dive are assumed to be zero offset correction errors, so
        depth observations at these extremes are interpolated between zero
        and the next observations when this occurs.

        Parameters
        ----------
        dive_model : {"unimodal", "smooth.spline"}
            Model to use for each dive for the purpose of dive phase
            identification.  One of `smooth.spline` or `unimodal`, to
            choose among smoothing spline or unimodal regression. For dives
            with less than five observations, smoothing spline regression
            is used regardless.
        smooth_par : float, optional
            Amount of smoothing when ``dive.model="smooth.spline"``. If it
            is `None`, then the smoothing parameter is determined by
            Generalized Cross-validation (GCV). Ignored with default
            ``dive.model="unimodal"``.
        knot_factor : int, optional
            Multiplier for the number of samples in the dive.  This is used
            to construct the time predictor for the derivative.
        descent_crit_q : float, optional
            Critical quantile of rates of descent below which descent is
            deemed to have ended.
        ascent_crit_q : float, optional
            Critical quantile of rates of ascent above which ascent is
            deemed to have started.

        Notes
        -----

        1. Unimodal method: in this default model, the spline is
        constrained to be unimodal [2]_, assuming the diver must return to
        the surface to breathe. The model is fitted using `R`'s `uniReg`
        package. This model and constraint are consistent with the
        definition of dives in air-breathers, so is appropriate for this
        group of divers. A major advantage of this approach over the next
        one is that the degree of smoothing is determined via restricted
        maximum likelihood, and has no influence on identifying the
        transition between descent and ascent. Therefore, unimodal
        regression splines make the latter transition clearer compared to
        using smoothing splines. Note that dives with less than five
        samples are fit using smoothing splines regardless, as they produce
        the same fit as unimodal regression but much faster. Therefore,
        ensure that the parameters for that model are appropriate for the
        data, although defaults are reasonable.

        2. Smooth spline: in this model, specified via
        ``dive_model="smooth.spline"``, a smoothing spline is used to model
        each dive, using the chosen smoothing parameter. Dive phases
        identified via this model, however, are highly sensitive to the
        degree of smoothing (`smooth_par`) used, thus making it difficult
        to determine what amount of smoothing is adequate.

        The first derivative of the spline is evaluated at a set of knots
        to calculate the vertical rate throughout the dive and determine
        the end of descent and beginning of ascent. This set of knots is
        established using a regular time sequence with beginning and end
        equal to the extremes of the input sequence, and with length equal
        to :math:`N \times knot\_factor`. Equivalent procedures are used
        for detecting descent and ascent phases.

        Once one of the models above has been fitted to each dive, the
        quantile corresponding to (`descent_crit_q`) of all the positive
        derivatives (rate of descent) at the beginning of the dive is used
        as threshold for determining the end of descent. Descent is deemed
        to have ended at the *first* minimum derivative, and the nearest
        input time observation is considered to indicate the end of
        descent. The sign of the comparisons is reversed for detecting the
        ascent. If observed depth to the left and right of the derivative
        defining the ascent are the same, the right takes precedence.

        The particular dive phase categories are subsequently defined using
        simple set operations.

        References
        ----------

        .. [2] Koellmann, C., Ickstadt, K. and Fried, R. (2014) Beyond
           unimodal regression: modelling multimodality with piecewise
           unimodal, mixture or additive regression. Technical Report
           8. `<https://sfb876.tu-dortmund.de/FORSCHUNG/techreports.html>`_,
           SFB 876, TU Dortmund

        Examples
        --------
        ZOC using the "offset" method for convenience

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd("TDRPhases")
        >>> tdrX.zoc("offset", offset=3)

        Detect wet/dry phases and dives with 3 m threshold

        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)

        Detect dive phases using the "unimodal" method and selected
        parameters

        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)

        """
        # Retrieve copy of depth from our own property
        depth = self.depth_zoc
        depth_py = depth.to_series()
        phases_df = self.get_dives_details("row_ids")
        dive_ids = self.get_dives_details("row_ids", columns="dive_id")
        ok = (dive_ids > 0) & ~depth_py.isna()
        xx = pd.Categorical(np.repeat(["X"], phases_df.shape[0]),
                            categories=["D", "DB", "B", "BA",
                                        "DA", "A", "X"])
        dive_phases = pd.Series(xx, index=phases_df.index)

        if any(ok):
            ddepths = depth_py[ok]  # diving depths
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

            cval_list = []
            spl_der_list = []
            spl_list = []
            for name, grp in grouped:
                res = _cut_dive(grp, dive_model=dive_model,
                                smooth_par=smooth_par,
                                knot_factor=knot_factor,
                                descent_crit_q=descent_crit_q,
                                ascent_crit_q=ascent_crit_q)
                dive_phases.loc[grp.index] = (res.pop("label_matrix")[:, 1])
                # Splines
                spl = res.pop("dive_spline")
                # Convert directly into a dict, with each element turned
                # into a list of R objects.  Access each via
                # `_get_dive_spline_slot`
                spl_dict = dict(zip(spl.names, list(spl)))
                spl_list.append(spl_dict)
                # Spline derivatives
                spl_der = res.pop("spline_deriv")
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

        # Update the `dives` attribute
        self.dives["row_ids"]["dive_phase"] = dive_phases
        (self.params["dives"]
         .update(dict(dive_model=dive_model, smooth_par=smooth_par,
                      knot_factor=knot_factor,
                      descent_crit_q=descent_crit_q,
                      ascent_crit_q=ascent_crit_q)))

        logger.info("Finished detecting dive phases")

    def get_dives_details(self, key, columns=None):
        """Accessor for the `dives` attribute

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
                raise KeyError("\'{}\' not available.".format(key))

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

        return odata

    def _get_wet_activity(self):
        return self._wet_dry

    wet_dry = property(_get_wet_activity)
    """Wet/dry activity labels

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: `phase_id` and `phase_label` for each
        measurement.

    """

    def get_phases_params(self, key):
        """Return parameters used for identifying wet/dry or diving phases.

        Parameters
        ----------
        key: {'wet_dry', 'dives'}

        Returns
        -------
        out : dict

        """
        try:
            params = self.params[key]
        except KeyError:
            msg = "key must be one of: {}".format(self.params.keys())
            logger.error(msg)
            raise KeyError(msg)
        return params

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
        idata = self.get_dives_details("splines")[diveNo]
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

        return odata

    def get_dive_deriv(self, diveNo, phase=None):
        """Retrieve depth spline derivative for a given dive

        Parameters
        ----------
        diveNo : int
            Dive number to retrieve derivative for.
        phase : {"descent", "bottom", "ascent"}
            If provided, the dive phase to retrieve data for.

        Returns
        -------
        out : pandas.DataFrame

        """
        der = self.get_dives_details("spline_derivs").loc[diveNo]
        crit_vals = self.get_dives_details("crit_vals").loc[diveNo]
        spl_data = self.get_dives_details("splines")[diveNo]["data"]
        spl_times = np.array(spl_data[0])  # x row is time steps in (s)

        if phase == "descent":
            descent_crit = int(crit_vals["descent_crit"])
            deltat_crit = pd.Timedelta(spl_times[descent_crit], unit="s")
            oder = der.loc[:deltat_crit]
        elif phase == "bottom":
            descent_crit = int(crit_vals["descent_crit"])
            deltat1 = pd.Timedelta(spl_times[descent_crit], unit="s")
            ascent_crit = int(crit_vals["ascent_crit"])
            deltat2 = pd.Timedelta(spl_times[ascent_crit], unit="s")
            oder = der[(der.index >= deltat1) & (der.index <= deltat2)]
        elif phase == "ascent":
            ascent_crit = int(crit_vals["ascent_crit"])
            deltat_crit = pd.Timedelta(spl_times[ascent_crit], unit="s")
            oder = der.loc[deltat_crit:]
        elif phase is None:
            oder = der
        else:
            msg = "`phase` must be 'descent', 'bottom' or 'ascent'"
            logger.error(msg)
            raise KeyError(msg)

        return oder

    def _get_dive_deriv_stats(self, diveNo):
        """Calculate stats for the depth derivative of a given dive

        """
        desc = self.get_dive_deriv(diveNo, "descent")
        bott = self.get_dive_deriv(diveNo, "bottom")
        asc = self.get_dive_deriv(diveNo, "ascent")
        # Rename DataFrame to match diveNo
        desc_sts = (pd.DataFrame(desc.describe().iloc[1:]).transpose()
                    .add_prefix("descD_").rename({"y": diveNo}))
        bott_sts = (pd.DataFrame(bott.describe().iloc[1:]).transpose()
                    .add_prefix("bottD_").rename({"y": diveNo}))
        asc_sts = (pd.DataFrame(asc.describe().iloc[1:]).transpose()
                   .add_prefix("ascD_").rename({"y": diveNo}))
        sts = pd.merge(desc_sts, bott_sts, left_index=True,
                       right_index=True)
        sts = pd.merge(sts, asc_sts, left_index=True, right_index=True)

        return sts

    def time_budget(self, ignore_z=True, ignore_du=True):
        """Summary of wet/dry activities at the broadest time scale

        Parameters
        ----------
        ignore_z : bool, optional
            Whether to ignore trivial aquatic periods.
        ignore_du : bool, optional
            Whether to ignore diving and underwater periods.

        Returns
        -------
        out : pandas.DataFrame
            DataFrame indexed by phase id, with categorical activity label
            for each phase, and beginning and ending times.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd("TDRPhases")
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.time_budget(ignore_z=True,
        ...                  ignore_du=True)  # doctest: +ELLIPSIS
                                 beg phase_label                 end
        phase_id
        1        2002-01-05      ...           L 2002-01-05      ...
        ...

        """
        phase_lab = self.wet_dry["phase_label"]
        idx_name = phase_lab.index.name
        labels = phase_lab.reset_index()
        if ignore_z:
            labels = labels.mask(labels == "Z", "L")
        if ignore_du:
            labels = labels.mask((labels == "U") | (labels == "D"), "W")

        grp_key = rle_key(labels["phase_label"]).rename("phase_id")
        labels_grp = labels.groupby(grp_key)

        begs = labels_grp.first().rename(columns={idx_name: "beg"})
        ends = labels_grp.last()[idx_name].rename("end")

        return pd.concat((begs, ends), axis=1)

    def stamp_dives(self, ignore_z=True):
        """Identify the wet activity phase corresponding to each dive

        Parameters
        ----------
        ignore_z : bool, optional
            Whether to ignore trivial aquatic periods.

        Returns
        -------
        out : pandas.DataFrame
            DataFrame indexed by dive ID, and three columns identifying
            which phase thy are in, and the beginning and ending time
            stamps.

        Examples
        --------
        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd("TDRPhases")
        >>> tdrX.zoc("offset", offset=3)
        >>> tdrX.detect_wet()
        >>> tdrX.detect_dives(3)
        >>> tdrX.detect_dive_phases("unimodal", descent_crit_q=0.01,
        ...                         ascent_crit_q=0, knot_factor=20)
        >>> tdrX.stamp_dives(ignore_z=True)  # doctest: +ELLIPSIS
                 phase_id                 beg                 end
        dive_id
        1               2 2002-01-05      ... 2002-01-06      ...

        """
        phase_lab = self.wet_dry["phase_label"]
        idx_name = phase_lab.index.name
        # "U" and "D" considered as "W" here
        phase_lab = phase_lab.mask(phase_lab.isin(["U", "D"]), "W")
        if ignore_z:
            phase_lab = phase_lab.mask(phase_lab == "Z", "L")

        dive_ids = self.get_dives_details("row_ids", columns="dive_id")

        grp_key = rle_key(phase_lab).rename("phase_id")

        isdive = dive_ids > 0
        merged = (pd.concat((grp_key, dive_ids, phase_lab), axis=1)
                  .loc[isdive, :].reset_index())
        # Rest index to use in first() and last()
        merged_grp = merged.groupby("phase_id")

        dives_ll = []
        for name, group in merged_grp:
            dives_uniq = pd.Series(group["dive_id"].unique(),
                                   name="dive_id")
            beg = [group[idx_name].iloc[0]] * dives_uniq.size
            end = [group[idx_name].iloc[-1]] * dives_uniq.size
            dive_df = pd.DataFrame({'phase_id': [name] * dives_uniq.size,
                                    'beg': beg,
                                    'end': end}, index=dives_uniq)
            dives_ll.append(dive_df)

        dives_all = pd.concat(dives_ll)
        return dives_all
