"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import pandas as pd
import scipy.optimize as scioptim
import skdiveMove.bouts as skbouts
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _genx(n, p, lda0, lda1):
    chooser = np.random.uniform(size=n)
    # We need to convert from scale to rate parameter
    proc1 = np.random.exponential(1 / lda0, size=n)
    proc2 = np.random.exponential(1 / lda1, size=n)
    proc_mix = np.where(chooser < p, proc1, proc2)
    return(proc_mix)


class TestBoutsMLE(ut.TestCase):
    """Test `Bouts` class methods

    """

    def setUp(self):
        """Initialize tests

        Attributes
        ----------
        x : ndarray
        pars_true : pandas.Series
        nsamp : int
        mle_opts1, mle_opts2 : dict

        """
        p_true = 0.7
        lda0_true = 0.05
        lda1_true = 0.005
        pars_true = pd.Series({"lambda0": lda0_true,
                               "lambda1": lda1_true,
                               "p": p_true})
        self.pars_true = pars_true
        nsamp = 10000
        x = _genx(nsamp,
                  pars_true.loc["p"],
                  pars_true.loc["lambda0"],
                  pars_true.loc["lambda1"])
        self.nsamp = nsamp
        self.x = x

        # Fixed bounds fit 1
        p_bnd = (-2, None)
        lda0_bnd = (-5, None)
        lda1_bnd = (-10, None)
        self.mle_opts1 = dict(method="L-BFGS-B",
                              bounds=(p_bnd, lda0_bnd, lda1_bnd))
        # Fixed bounds fit 2
        p_bnd = (1e-1, None)
        lda0_bnd = (1e-3, None)
        lda1_bnd = (1e-6, None)
        self.mle_opts2 = dict(method="L-BFGS-B",
                              bounds=(p_bnd, lda0_bnd, lda1_bnd))

    def test_fit(self):
        x = self.x
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle_opts1
        fit2_opts = self.mle_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        self.assertIsInstance(fit1, scioptim.OptimizeResult)
        self.assertIsInstance(fit2, scioptim.OptimizeResult)

        self.assertTrue(fit1.success)
        self.assertTrue(fit2.success)

        self.assertEqual(fit1.x.size, 3)
        self.assertEqual(fit2.x.size, 3)

    def test_bec(self):
        x = self.x
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        x = self.x
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle_opts1
        fit2_opts = self.mle_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        bec = xbouts.bec(fit2)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 1)

    def test_plot_fit(self):
        x = self.x
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle_opts1
        fit2_opts = self.mle_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

    def test_plot_ecdf(self):
        x = self.x
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle_opts1
        fit2_opts = self.mle_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()


if __name__ == '__main__':
    ut.main()
