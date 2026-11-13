"""Test BoutsMLE class

"""

import numpy as np
import pandas as pd
import scipy.optimize as scioptim
import skdiveMove.bouts as skbouts
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
import pytest                                # noqa: E402

from skdiveMove.core import robjs, diveMove  # noqa: E402
from skdiveMove.tests.scenarios import BoutsBaseScenario  # noqa: E402


class MLEScenario(BoutsBaseScenario):
    """Testing scenario for BoutsMLE

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
    mle_opts1, mle_opts2 : dict
        Dictionary with keys and values passed to optimization routine for
        first and second fits.

    """
    def __init__(self, nsamp, pars_true):
        super().__init__(nsamp)

        npars = len(pars_true)
        if npars == 3:
            self.x = self.generate_proc2(pars_true)
            # Fixed bounds fit 1
            p_bnd = (-2, None)
            lda0_bnd = (-5, None)
            lda1_bnd = (-10, None)
            self.mle_opts1 = dict(method="L-BFGS-B",
                                  bounds=(p_bnd, lda0_bnd, lda1_bnd))
            # Fixed bounds fit 2
            p_bnd = (1e-2, None)
            lda0_bnd = (1e-4, None)
            lda1_bnd = (1e-8, None)
            self.mle_opts2 = dict(method="L-BFGS-B",
                                  bounds=(p_bnd, lda0_bnd, lda1_bnd))
        elif npars == 5:
            self.x = self.generate_proc3(pars_true)
            # Fixed bounds fit 1
            p0_bnd = (-5, None)
            p1_bnd = (-5, None)
            lda0_bnd = (-6, None)
            lda1_bnd = (-8, None)
            lda2_bnd = (-12, None)
            self.mle_opts1 = dict(method="L-BFGS-B",
                                  bounds=(p0_bnd, p1_bnd,
                                          lda0_bnd, lda1_bnd, lda2_bnd))
            # Fixed bounds fit 2
            p0_bnd = (1e-3, 9.9e-1)
            p1_bnd = (1e-3, 9.9e-1)
            lda0_bnd = (2e-2, 1e-1)
            lda1_bnd = (3e-3, 5e-2)
            lda2_bnd = (1e-5, 1e-3)
            self.mle_opts2 = dict(method="L-BFGS-B",
                                  bounds=(p0_bnd, p1_bnd,
                                          lda0_bnd, lda1_bnd, lda2_bnd))
        else:
            raise ValueError("`pars_true` must have 3 or 5 values.")

        self.pars_true = pars_true


@pytest.fixture(scope="module")
def bouts2():
    """Generate test instance for 2-process scenario."""
    nsamp = 10000
    p_true = 0.7
    lda0_true = 0.05
    lda1_true = 0.005
    pars_true = pd.Series({"lambda0": lda0_true,
                           "lambda1": lda1_true,
                           "p": p_true})

    return MLEScenario(nsamp, pars_true)


@pytest.fixture(scope="module")
def bouts3():
    """Generate test instance for 3-process scenario."""
    nsamp = 10000
    p0_true = 0.6           # fast
    p1_true = 0.7           # prop of medium to (medium + slow) procs
    lda0_true = 0.05
    lda1_true = 0.01
    lda2_true = 8e-4
    pars_true = pd.Series({"lambda0": lda0_true,
                           "lambda1": lda1_true,
                           "lambda2": lda2_true,
                           "p0": p0_true,
                           "p1": p1_true})

    return MLEScenario(nsamp, pars_true)


def test_fit(bouts2, bouts3):
    # Two process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsMLE(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    fit1_opts = bouts2.mle_opts1
    fit2_opts = bouts2.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)
    assert isinstance(fit1, scioptim.OptimizeResult)
    assert isinstance(fit2, scioptim.OptimizeResult)

    assert fit1.success
    assert fit2.success

    assert fit1.x.size == 3
    assert fit2.x.size == 3

    # Three process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsMLE(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    fit1_opts = bouts3.mle_opts1
    fit2_opts = bouts3.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)
    assert isinstance(fit1, scioptim.OptimizeResult)
    assert isinstance(fit2, scioptim.OptimizeResult)

    assert fit1.success
    assert fit2.success

    assert fit1.x.size == 5
    assert fit2.x.size == 5

    # loglik_fun expected failures
    pars_bad = np.ones(6)
    with pytest.raises(KeyError):
        xbouts.negMLEll(pars_bad, x3)


def test_bec(bouts2, bouts3):
    # Two process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsMLE(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    fit1_opts = bouts2.mle_opts1
    fit2_opts = bouts2.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)
    bec = xbouts.bec(fit2)
    assert isinstance(bec, np.ndarray)
    assert bec.size == 1

    # Three process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsMLE(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    fit1_opts = bouts3.mle_opts1
    fit2_opts = bouts3.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)
    bec = xbouts.bec(fit2)
    assert isinstance(bec, np.ndarray)
    assert bec.size == 2


@pytest.mark.plotting
def test_plot_fit(bouts2, bouts3):
    # Two process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsMLE(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    fit1_opts = bouts2.mle_opts1
    fit2_opts = bouts2.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)

    fig, ax = plt.subplots()
    _ = xbouts.plot_fit(fit2, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2
    plt.close()
    # Without Axes
    _ = xbouts.plot_fit(fit2)
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2

    # Three process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsMLE(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    fit1_opts = bouts3.mle_opts1
    fit2_opts = bouts3.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)

    fig, ax = plt.subplots()
    _ = xbouts.plot_fit(fit2, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2


@pytest.mark.plotting
def test_plot_ecdf(bouts2, bouts3):
    # Two process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsMLE(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    fit1_opts = bouts2.mle_opts1
    fit2_opts = bouts2.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)

    fig, ax = plt.subplots()
    _ = xbouts.plot_ecdf(fit2, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2
    plt.close()
    # Without Axes
    _ = xbouts.plot_ecdf(fit2)
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 2

    # Three process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsMLE(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    fit1_opts = bouts3.mle_opts1
    fit2_opts = bouts3.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)

    fig, ax = plt.subplots()
    _ = xbouts.plot_ecdf(fit2, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2


def test_compare2r(bouts2, bouts3):
    # Two process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsMLE(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    fit1_opts = bouts2.mle_opts1
    fit2_opts = bouts2.mle_opts2
    fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                            fit2_opts=fit2_opts)

    robjs.r("""
         opts0 <- list(method="L-BFGS-B", lower=c(-2, -5, -10))
         ## opts1 <- list(method="L-BFGS-B", lower=c(1e-1, 1e-3, 1e-6))
         opts1 <- list(method="L-BFGS-B", lower=c(1e-2, 1e-4, 1e-8))
    """)

    x_r = robjs.FloatVector(x2)
    xbouts_r = diveMove.boutfreqs(x_r, bw=5, plot=False)
    startval = diveMove.boutinit(xbouts_r, robjs.FloatVector([80]),
                                 plot=False)
    fit_r = diveMove.fitMLEbouts(xbouts_r, start=startval,
                                 optim_opts0=robjs.r["opts0"],
                                 optim_opts1=robjs.r["opts1"])
    coefs_r = np.array(robjs.r("coef")(fit_r))
    np.testing.assert_allclose(coefs_r, fit2.x, rtol=1e-4)
