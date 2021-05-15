"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import pandas as pd
import skdiveMove.bouts as skbouts
from skdiveMove.bouts.bouts import ecdf
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


class TestBoutsNLS(ut.TestCase):
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
        nls3_opts : dict
            Options for fitting the NLS model in the three-process case.

        """
        # Two-process case
        p_true = 0.7
        lda0_true = 0.05
        lda1_true = 0.005
        pars_true = pd.Series({"lambda0": lda0_true,
                               "lambda1": lda1_true,
                               "p": p_true})
        cls.pars_true = pars_true
        nsamp = 10000
        rng = np.random.default_rng()
        x = skbouts.random_mixexp(nsamp,
                                  pars_true.loc["p"],
                                  (pars_true.loc[["lambda0", "lambda1"]]
                                   .to_numpy()), rng=rng)
        cls.nsamp = nsamp
        cls.x2 = pd.Series(x)

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
        # Bounds for NLS fit; flattened, two per process (a, lambda).
        # Two-tuple with lower and upper bounds for each parameter.
        nls_opts = dict(bounds=(([100, 1e-3, 100, 1e-3, 100, 1e-6]),
                                ([5e4, 1, 5e4, 1, 5e4, 1])))
        cls.nls3_opts = nls_opts

    def test_init(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

        xbouts = skbouts.BoutsNLS(x, 0.25, method="seq_diff")
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

        xbouts = skbouts.BoutsNLS(x, 0.25, method="seq_diff")
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

    def test_str(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIn("Class BoutsNLS object", xbouts.__str__())
        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIn("Class BoutsNLS object", xbouts.__str__())

    def test_init_pars(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)

        init_pars = xbouts.init_pars([80], plot=False)
        self.assertIsInstance(init_pars, pd.DataFrame)
        self.assertEqual(init_pars.size, 4)

        fig, ax = plt.subplots()
        _ = xbouts.init_pars([80], plot=True, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 3)
        plt.close()
        # Without Axes
        _ = xbouts.init_pars([80], plot=True)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 3)
        plt.close()

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)

        init_pars = xbouts.init_pars([75, 220], plot=False)
        self.assertIsInstance(init_pars, pd.DataFrame)
        self.assertEqual(init_pars.size, 6)

        fig, ax = plt.subplots()
        _ = xbouts.init_pars([75, 220], plot=True, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 4)
        plt.close()

        # Not implemented
        # ---------------
        self.assertRaises(IndexError, xbouts.init_pars,
                          [75, 220, 500], plot=False)

    def test_fit(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        self.assertIsInstance(coefs, pd.DataFrame)
        self.assertEqual(coefs.size, 4)
        self.assertEqual(pcov.size, coefs.size ** 2)

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        self.assertIsInstance(coefs, pd.DataFrame)
        self.assertEqual(coefs.size, 6)
        self.assertEqual(pcov.size, coefs.size ** 2)

    def test_bec(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 1)

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 2)

    def test_ecdf(self):
        x = self.x3
        p = self.pars_true3.loc["p0":].tolist()
        lda = self.pars_true3.loc["lambda0":"lambda2"]
        lda["lambda3"] = 0.1
        # Not implemented
        self.assertRaises(KeyError, ecdf, x, p, lda)

    def test_label_bouts(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        # label_bouts takes Series
        xlabeled = skbouts.label_bouts(pd.Series(x), bec)
        self.assertEqual(xlabeled.shape, x.shape)

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        xlabeled = skbouts.label_bouts(pd.Series(x), bec)
        self.assertEqual(xlabeled.shape, x.shape)

    def test_plot_fit(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(coefs, ax=ax)
        line = ax.get_lines()
        self.assertEqual(len(line), 1)
        plt.close()
        # Without Axes
        _ = xbouts.plot_fit(coefs)
        line = ax.get_lines()
        self.assertEqual(len(line), 1)
        plt.close()

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(coefs, ax=ax)
        line = ax.get_lines()
        self.assertEqual(len(line), 1)
        plt.close()

    def test_plot_ecdf(self):
        # Two-process
        # -----------
        x = self.x2
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(coefs, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()

        # Three-process
        # -------------
        x = self.x3
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([75, 220], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(coefs, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()


if __name__ == '__main__':
    ut.main()
