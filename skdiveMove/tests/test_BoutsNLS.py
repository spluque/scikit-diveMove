"""Test BoutsNLS class

"""

import numpy as np
import pandas as pd
import skdiveMove.bouts as skbouts
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest                    # noqa: E402

from skdiveMove.bouts.bouts import ecdf  # noqa: E402
from skdiveMove.tests.scenarios import BoutsBaseScenario  # noqa: E402


class NLSScenario(BoutsBaseScenario):
    """Testing scenario for BoutsNLS

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
    nls_opts : dict
        Dictionary with keys and values passed to optimization routine.

    """
    def __init__(self, nsamp, pars_true):
        super().__init__(nsamp)

        npars = len(pars_true)
        if npars == 3:
            self.x = self.generate_proc2(pars_true)
        elif npars == 5:
            self.x = self.generate_proc3(pars_true)
        else:
            raise ValueError("`pars_true` must have 3 or 5 values.")

        self.pars_true = pars_true
        # Bounds for NLS fit; flattened, two per process (a, lambda).
        # Two-tuple with lower and upper bounds for each parameter.
        nls_opts = dict(bounds=(([100, 1e-3, 100, 1e-3, 100, 1e-6]),
                                ([5e4, 1, 5e4, 1, 5e4, 1])))
        self.nls_opts = nls_opts


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
    return NLSScenario(nsamp, pars_true)


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
    return NLSScenario(nsamp, pars_true)


def test_init(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    assert isinstance(xbouts, skbouts.BoutsNLS)
    assert isinstance(xbouts.lnfreq, pd.DataFrame)

    xbouts = skbouts.BoutsNLS(x2, 0.25, method="seq_diff")
    assert isinstance(xbouts, skbouts.BoutsNLS)
    assert isinstance(xbouts.lnfreq, pd.DataFrame)

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    assert isinstance(xbouts, skbouts.BoutsNLS)
    assert isinstance(xbouts.lnfreq, pd.DataFrame)

    xbouts = skbouts.BoutsNLS(x3, 0.25, method="seq_diff")
    assert isinstance(xbouts, skbouts.BoutsNLS)
    assert isinstance(xbouts.lnfreq, pd.DataFrame)


def test_str(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    assert "Class BoutsNLS object" in xbouts.__str__()
    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    assert "Class BoutsNLS object" in xbouts.__str__()


@pytest.mark.plotting
def test_init_pars(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)

    init_pars = xbouts.init_pars([80], plot=False)
    assert isinstance(init_pars, pd.DataFrame)
    assert init_pars.size == 4

    fig, ax = plt.subplots()
    _ = xbouts.init_pars([80], plot=True, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 3
    plt.close()
    # Without Axes
    _ = xbouts.init_pars([80], plot=True)
    ax = plt.gca()
    lines = ax.get_lines()
    assert len(lines) == 3

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)

    init_pars = xbouts.init_pars([75, 220], plot=False)
    assert isinstance(init_pars, pd.DataFrame)
    assert init_pars.size == 6

    fig, ax = plt.subplots()
    _ = xbouts.init_pars([75, 220], plot=True, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 4
    plt.close()

    # Not implemented
    # ---------------
    with pytest.raises(IndexError):
        xbouts.init_pars([75, 220, 500], plot=False)


def test_fit(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    assert isinstance(coefs, pd.DataFrame)
    assert coefs.size == 4
    assert pcov.size == (coefs.size ** 2)

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    assert isinstance(coefs, pd.DataFrame)
    assert coefs.size == 6
    assert pcov.size == (coefs.size ** 2)


def test_bec(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    bec = xbouts.bec(coefs)
    assert isinstance(bec, np.ndarray)
    assert bec.size == 1

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    bec = xbouts.bec(coefs)
    assert isinstance(bec, np.ndarray)
    assert bec.size == 2


def test_ecdf(bouts3):
    x = bouts3.x
    p = bouts3.pars_true.loc["p0":].tolist()
    lda = bouts3.pars_true.loc["lambda0":"lambda2"]
    lda["lambda3"] = 0.1
    # Not implemented
    with pytest.raises(KeyError):
        ecdf(x, p, lda)


def test_label_bouts(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    bec = xbouts.bec(coefs)
    # label_bouts takes Series
    xlabeled = skbouts.label_bouts(pd.Series(x2), bec)
    assert xlabeled.shape == x2.shape

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    coefs, pcov = xbouts.fit(init_pars)
    bec = xbouts.bec(coefs)
    xlabeled = skbouts.label_bouts(pd.Series(x3), bec)
    assert xlabeled.shape == x3.shape


@pytest.mark.plotting
def test_plot_fit(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    coefs, pcov = xbouts.fit(init_pars)

    fig, ax = plt.subplots()
    _ = xbouts.plot_fit(coefs, ax=ax)
    line = ax.get_lines()
    assert len(line) == 1
    plt.close()
    # Without Axes
    _ = xbouts.plot_fit(coefs)
    line = ax.get_lines()
    assert len(line) == 1

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    coefs, pcov = xbouts.fit(init_pars)

    fig, ax = plt.subplots()
    _ = xbouts.plot_fit(coefs, ax=ax)
    line = ax.get_lines()
    assert len(line) == 1


@pytest.mark.plotting
def test_plot_ecdf(bouts2, bouts3):
    # Two-process
    # -----------
    x2 = bouts2.x
    xbouts = skbouts.BoutsNLS(x2, 5)
    init_pars = xbouts.init_pars([80], plot=False)

    coefs, pcov = xbouts.fit(init_pars)

    fig, ax = plt.subplots()
    _ = xbouts.plot_ecdf(coefs, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2

    # Three-process
    # -------------
    x3 = bouts3.x
    xbouts = skbouts.BoutsNLS(x3, 5)
    init_pars = xbouts.init_pars([75, 220], plot=False)

    coefs, pcov = xbouts.fit(init_pars)

    fig, ax = plt.subplots()
    _ = xbouts.plot_ecdf(coefs, ax=ax)
    lines = ax.get_lines()
    assert len(lines) == 2
