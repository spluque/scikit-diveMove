"""Meta classes for flexible testing of TDR and bout analysis classes

"""
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from skdiveMove.tests.get_sample_data import (
    diveMove2skd, random_mixexp)  # noqa: E402


class TDRScenario:
    """Test `TDR` class.

    Parameters
    ----------
    oclass : {"TDRSource", "ZOC", "TDRPhases", "TDR"}
        The class to retrieve from sample data.
    **kwargs
        Optional keyword arguments passed to class instantiator.

    Attributes
    ----------
    tdrX : TDR
        TDR instance.
    default_pars : dict
        Calibrate parameters.
    tdr_calib : TDRPhases
        Calibrated TDR instance.
    diveNo_seq : ndarray of shape (N,)
        Sequence of dive numbers in the time series.

    """
    def __init__(self, oclass, **kwargs) -> None:
        # An instance to work with
        self.tdrX = diveMove2skd(oclass, **kwargs)
        zoc_offset = 3
        dry_thr = 70
        wet_thr = 3610
        dive_thr = 3
        dive_model = "unimodal"
        smooth_par = 0.1
        knot_factor = 3
        descent_crit_q = 0
        ascent_crit_q = 0
        self.default_pars = {"zoc_offset": zoc_offset,
                             "dry_thr": dry_thr,
                             "wet_thr": wet_thr,
                             "dive_thr": dive_thr,
                             "dive_model": dive_model,
                             "smooth_par": smooth_par,
                             "knot_factor": knot_factor,
                             "descent_crit_q": descent_crit_q,
                             "ascent_crit_q": ascent_crit_q}
        self.diveNo_seq = None

    def zoc(self):
        tdrX = self.tdrX
        default_pars = self.default_pars
        tdrX.zoc(method="offset", offset=default_pars["zoc_offset"])

    def detect_wet(self):
        default_pars = self.default_pars
        self.zoc()
        self.tdrX.detect_wet(dry_thr=default_pars["dry_thr"],
                             wet_thr=default_pars["wet_thr"])

    def detect_dives(self):
        default_pars = self.default_pars
        self.detect_wet()
        self.tdrX.detect_dives(dive_thr=default_pars["dive_thr"])
        diveNo_max = (self.tdrX.get_dives_details("row_ids", "dive_id")
                      .max())
        self.diveNo_seq = np.arange(diveNo_max) + 1

    def detect_dive_phases(self):
        default_pars = self.default_pars
        self.detect_dives()
        self.tdrX.detect_dive_phases(
            dive_model=default_pars["dive_model"],
            smooth_par=default_pars["smooth_par"],
            knot_factor=default_pars["knot_factor"],
            descent_crit_q=default_pars["descent_crit_q"],
            ascent_crit_q=default_pars["ascent_crit_q"])


class BoutsBaseScenario(ABC):
    """Common base scenario for testing

    Subclasses must implement their own __init__ method.

    Attributes
    ----------
    nsamp : int
        Number of samples to draw from the mixed distributions.
    x : array_like
        1D array with samples from mixed distribution.
    pars_true : dict
        Dictionary with true parameters used to generate `x`, with keys:
        `p`, `lambda0`, and `lambda1`, if it is a 2-process scenario, or
        `p0`, `p1`, `lambda0`, `lambda1`, and `lambda2`, if it is a
        3-process scenario.
    nls_opts, mle_opts1, mle_opts2 : dict
        Dictionary with keys and values passed to optimization routine for
        first and second fits.

    """

    @abstractmethod
    def __init__(self, nsamp: int) -> None:
        self.nsamp = nsamp
        self.rng = np.random.default_rng()
        self.pars_true = None
        self.nls_opts = None
        self.mle_opts1 = None
        self.mle_opts2 = None
        self.x = None

    def generate_proc2(self, pars_true):
        # Generate samples for 2-process case
        x2 = random_mixexp(self.nsamp,
                           pars_true.loc["p"],
                           (pars_true.loc[["lambda0", "lambda1"]]
                            .to_numpy()), rng=self.rng)

        return pd.Series(x2)

    def generate_proc3(self, pars_true):
        # Generate samples for 3-process case
        x3 = random_mixexp(self.nsamp,
                           pars_true.loc[["p0", "p1"]].to_numpy(),
                           (pars_true.loc[
                               ["lambda0", "lambda1", "lambda2"]]
                            .to_numpy()), rng=self.rng)
        return pd.Series(x3)
