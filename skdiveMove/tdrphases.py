"""Class handling all TDR phase operations

Phase identification methods take `depth` as input.

Class and Methods Summary
-------------------------

.. autosummary::

   TDRPhases.detect_wet
   TDRPhases.detect_dives
   TDRPhases.detect_dive_phases
   TDRPhases.get_dives_details
   TDRPhases.get_wet_activity
   TDRPhases.get_params
   TDRPhases.time_budget
   TDRPhases.stamp_dives

"""

import logging
import numpy as np
import pandas as pd
from skdiveMove.core import robjs, cv, pandas2ri
from skdiveMove.helpers import get_var_sampling_interval

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


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

        lmtx = (robjs.r.slot(dmodel, dmodel_slots[0])
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


class TDRPhases:
    """Core TDR phase identification routines

    Attributes
    ----------
    wet_dry : pandas.DataFrame
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
    def __init__(self):
        self.wet_dry = None
        self.dives = dict(row_ids=None, model=None, splines=None,
                          spline_derivs=None, crit_vals=None)
        self.params = dict(wet_dry={}, dives={})

    def detect_wet(self, depth, dry_thr=70, wet_cond=None,
                   wet_thr=3610, interp_wet=False):
        """Detect wet/dry activity phases

        Set the ``wet_dry`` attribute.

        Parameters
        ----------
        depth : xarray.DataArray
            DataArray with zero-offset corrected depth measurements.
        dry_thr : float, optional
        wet_cond : bool mask, optional
        wet_thr : float, optional
        interp_wet : bool, optional

        Notes
        -----

        See details for arguments in diveMove's ``calibrateDepth``.  Unlike
        `diveMove`, the beginning/ending times for each phase are not
        stored with the class instance, as this information can be
        retrieved via the `.time_budget` method.

        """
        depth_py = depth.to_series()
        time_py = depth_py.index
        dtime = get_var_sampling_interval(depth).total_seconds()

        if wet_cond:
            wet_cond = (pd.Series(wet_cond, index=time_py)
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
                                  index=time_py)

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
        self.wet_dry = phases
        wet_dry_params = dict(dry_thr=dry_thr, wet_thr=wet_thr,
                              interp_wet=interp_wet)
        self.params["wet_dry"].update(wet_dry_params)

    def detect_dives(self, depth, dive_thr):
        """Identify dive events

        Set the ``dives`` attribute's "row_ids" dictionary element, and
        update the ``wet_act`` attribute's "phases" dictionary element.

        Parameters
        ----------
        depth : xarray.DataArray
            DataArray with zero-offset corrected depth measurements.
        dive_thr : float

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        """
        depth_py = depth.to_series()
        act_phases = self.wet_dry["phase_label"]
        detDiveFun = robjs.r("""detDiveFun <- diveMove:::.detDive""")
        with cv.localconverter(robjs.default_converter +
                               pandas2ri.converter):
            phases_df = detDiveFun(pd.Series(depth_py),
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
        self.wet_dry["phase_label"] = dive_activity
        self.params["dives"].update({'dive_thr': dive_thr})

    def detect_dive_phases(self, depth, dive_model, smooth_par=0.1,
                           knot_factor=3, descent_crit_q=0,
                           ascent_crit_q=0):
        """Detect dive phases

        Complete filling the ``dives`` attribute.

        Parameters
        ----------
        depth : xarray.DataArray
            DataArray with zero-offset corrected depth measurements.
        dive_model : {"unimodal", "smooth.spline"}
        smooth_par : float, optional
        knot_factor : int, optional
        descent_crit_q : float, optional
        ascent_crit_q : float, optional

        Notes
        -----
        See details for arguments in diveMove's ``calibrateDepth``.

        """
        depth_py = depth.to_series()
        phases_df = self.get_dives_details("row_ids")
        dive_ids = self.get_dives_details("row_ids", columns="dive_id")
        ok = (dive_ids > 0) & ~depth_py.isna()

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

            xx = pd.Categorical(np.repeat(["X"], phases_df.shape[0]),
                                categories=["D", "DB", "B", "BA",
                                            "DA", "A", "X"])
            self.dives["row_ids"]["dive_phase"] = xx
            dive_phases = self.dives["row_ids"]["dive_phase"]
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

        (self.params["dives"]
         .update(dict(dive_model=dive_model, smooth_par=smooth_par,
                      knot_factor=knot_factor,
                      descent_crit_q=descent_crit_q,
                      ascent_crit_q=ascent_crit_q)))

    def get_dives_details(self, key, columns=None):
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

    def get_wet_activity(self):
        """Accessor for the ``wet_dry`` attribute

        """
        return(self.wet_dry)

    def get_params(self, key):
        """Return parameters used for identifying wet/dry or diving phases.

        Parameters
        ----------
        key: {'wet_dry', 'dives'}

        """
        if key == "wet_dry":
            return(self.params["wet_dry"])
        elif key == "dives":
            return(self.params["dives"])

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

        return(odata)

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

        """
        labels = (self.get_wet_activity()["phase_label"]
                  .reset_index())
        if ignore_z:
            labels = labels.mask(labels == "Z", "L")
        if ignore_du:
            labels = labels.mask((labels == "U") | (labels == "D"), "W")

        grp_key = (labels["phase_label"]
                   .ne(labels["phase_label"].shift())
                   .cumsum() + 1).rename("phase_id")
        labels_grp = labels.groupby(grp_key)

        begs = labels_grp.first().rename(columns={"date_time": "beg"})
        ends = labels_grp.last()["date_time"].rename("end")

        return(pd.concat((begs, ends), axis=1))

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

        """
        phase_lab = self.get_wet_activity()["phase_label"]
        # "U" and "D" considered as "W" here
        phase_lab = phase_lab.mask(phase_lab.isin(["U", "D"]), "W")
        if ignore_z:
            phase_lab = phase_lab.mask(phase_lab == "Z", "L")

        dive_ids = self.get_dives_details("row_ids", columns="dive_id")

        grp_key = (phase_lab
                   .ne(phase_lab.shift())
                   .cumsum() + 1).rename("phase_id")

        isdive = dive_ids > 0
        merged = (pd.concat((grp_key, dive_ids, phase_lab), axis=1)
                  .loc[isdive, :].reset_index())
        # Rest index to use in first() and last()
        merged_grp = merged.groupby("phase_id")

        dives_ll = []
        for name, group in merged_grp:
            dives_uniq = pd.Series(group["dive_id"].unique(),
                                   name="dive_id")
            beg = [group["date_time"].iloc[0]] * dives_uniq.size
            end = [group["date_time"].iloc[-1]] * dives_uniq.size
            dive_df = pd.DataFrame({'phase_id': [name] * dives_uniq.size,
                                    'beg': beg,
                                    'end': end}, index=dives_uniq)
            dives_ll.append(dive_df)

        dives_all = pd.concat(dives_ll)
        return(dives_all)