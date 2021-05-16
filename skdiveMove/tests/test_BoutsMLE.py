"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import pandas as pd
import scipy.optimize as scioptim
import skdiveMove.bouts as skbouts
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from skdiveMove.core import robjs, diveMove  # noqa: E402


class TestBoutsMLE(ut.TestCase):
    """Test `Bouts` class methods

    """

    @classmethod
    def setUpClass(cls):
        """Initialize tests

        Attributes
        ----------
        x2 : ndarray
            Random samples from a two-process mixture of exponential
            distributions.
        x3 : ndarray
            Random samples from a three-process mixture of exponential
            distributions.
        pars_true2 : pandas.Series
            True parameters for generating `x2`.
        pars_true3 : pandas.Series
            True parameters for generating `x3`.
        nsamp : int
            Number of samples in `x2` and `x3`.
        mle2_opts1, mle2_opts2 : dict
            Options for the first and second fits in the two-process case.
        mle3_opts1, mle3_opts2 : dict
            Options for the first and second fits in the three-process case.

        """
        # Two-process case
        # ----------------
        p_true = 0.7
        lda0_true = 0.05
        lda1_true = 0.005
        pars_true = pd.Series({"lambda0": lda0_true,
                               "lambda1": lda1_true,
                               "p": p_true})
        cls.pars_true2 = pars_true
        nsamp = 10000
        rng = np.random.default_rng()
        x = skbouts.random_mixexp(nsamp, pars_true.loc["p"],
                                  (pars_true.loc[["lambda0", "lambda1"]]
                                   .to_numpy()), rng=rng)
        cls.nsamp = nsamp
        cls.x2 = x
        # Fixed bounds fit 1
        p_bnd = (-2, None)
        lda0_bnd = (-5, None)
        lda1_bnd = (-10, None)
        cls.mle2_opts1 = dict(method="L-BFGS-B",
                              bounds=(p_bnd, lda0_bnd, lda1_bnd))
        # Fixed bounds fit 2
        p_bnd = (1e-2, None)
        lda0_bnd = (1e-4, None)
        lda1_bnd = (1e-8, None)
        cls.mle2_opts2 = dict(method="L-BFGS-B",
                              bounds=(p_bnd, lda0_bnd, lda1_bnd))

        # Three-process case
        # ------------------
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
        cls.pars_true3 = pars_true
        x = skbouts.random_mixexp(nsamp, pars_true.loc[["p0", "p1"]],
                                  (pars_true.loc[["lambda0", "lambda1",
                                                  "lambda2"]]
                                   .to_numpy()), rng=rng)
        cls.x3 = x
        # Fixed bounds fit 1
        p0_bnd = (-5, None)
        p1_bnd = (-5, None)
        lda0_bnd = (-6, None)
        lda1_bnd = (-8, None)
        lda2_bnd = (-12, None)
        cls.mle3_opts1 = dict(method="L-BFGS-B",
                              bounds=(p0_bnd, p1_bnd,
                                      lda0_bnd, lda1_bnd, lda2_bnd))
        # Fixed bounds fit 2
        p0_bnd = (1e-3, 9.9e-1)
        p1_bnd = (1e-3, 9.9e-1)
        lda0_bnd = (2e-2, 1e-1)
        lda1_bnd = (3e-3, 5e-2)
        lda2_bnd = (1e-5, 1e-3)
        cls.mle3_opts2 = dict(method="L-BFGS-B",
                              bounds=(p0_bnd, p1_bnd,
                                      lda0_bnd, lda1_bnd, lda2_bnd))

    def test_fit(self):
        # Two process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle2_opts1
        fit2_opts = self.mle2_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        self.assertIsInstance(fit1, scioptim.OptimizeResult)
        self.assertIsInstance(fit2, scioptim.OptimizeResult)

        self.assertTrue(fit1.success)
        self.assertTrue(fit2.success)

        self.assertEqual(fit1.x.size, 3)
        self.assertEqual(fit2.x.size, 3)

        # Three process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        fit1_opts = self.mle3_opts1
        fit2_opts = self.mle3_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        self.assertIsInstance(fit1, scioptim.OptimizeResult)
        self.assertIsInstance(fit2, scioptim.OptimizeResult)

        self.assertTrue(fit1.success)
        self.assertTrue(fit2.success)

        self.assertEqual(fit1.x.size, 5)
        self.assertEqual(fit2.x.size, 5)

        # loglik_fun expected failures
        pars_bad = np.ones(6)
        self.assertRaises(KeyError, xbouts.negMLEll, pars_bad, x)

    def test_bec(self):
        # Two process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle2_opts1
        fit2_opts = self.mle2_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        bec = xbouts.bec(fit2)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 1)

        # Three process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        fit1_opts = self.mle3_opts1
        fit2_opts = self.mle3_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)
        bec = xbouts.bec(fit2)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 2)

    def test_plot_fit(self):
        # Two process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle2_opts1
        fit2_opts = self.mle2_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()
        # Without Axes
        _ = xbouts.plot_fit(fit2)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

        # Three process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        fit1_opts = self.mle3_opts1
        fit2_opts = self.mle3_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

    def test_plot_ecdf(self):
        # Two process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle2_opts1
        fit2_opts = self.mle2_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()
        # Without Axes
        _ = xbouts.plot_ecdf(fit2)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

        # Three process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        fit1_opts = self.mle3_opts1
        fit2_opts = self.mle3_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(fit2, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

    def test_compare2r(self):
        # Two process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsMLE(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        fit1_opts = self.mle2_opts1
        fit2_opts = self.mle2_opts2
        fit1, fit2 = xbouts.fit(init_pars, fit1_opts=fit1_opts,
                                fit2_opts=fit2_opts)

        robjs.r("""
             opts0 <- list(method="L-BFGS-B", lower=c(-2, -5, -10))
             ## opts1 <- list(method="L-BFGS-B", lower=c(1e-1, 1e-3, 1e-6))
             opts1 <- list(method="L-BFGS-B", lower=c(1e-2, 1e-4, 1e-8))
        """)

        x_r = robjs.FloatVector(x)
        xbouts_r = diveMove.boutfreqs(x_r, bw=5, plot=False)
        startval = diveMove.boutinit(xbouts_r, robjs.FloatVector([80]),
                                     plot=False)
        fit_r = diveMove.fitMLEbouts(xbouts_r, start=startval,
                                     optim_opts0=robjs.r["opts0"],
                                     optim_opts1=robjs.r["opts1"])
        coefs_r = np.array(robjs.r("coef")(fit_r))
        np.testing.assert_allclose(coefs_r, fit2.x, rtol=1e-4)


if __name__ == '__main__':
    ut.main()
