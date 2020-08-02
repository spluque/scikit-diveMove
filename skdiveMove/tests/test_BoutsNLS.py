"""Unit test for TDR class

"""

import unittest as ut
import numpy as np
import pandas as pd
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


class TestBoutsNLS(ut.TestCase):
    """Test `Bouts` class methods

    """

    def setUp(self):
        """Initialize tests

        Attributes
        ----------
        x : pandas.Series
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
        self.x = pd.Series(x)

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

    def test_init(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

        xbouts = skbouts.BoutsNLS(x, 0.25, method="seq_diff")
        self.assertIsInstance(xbouts, skbouts.BoutsNLS)
        self.assertIsInstance(xbouts.lnfreq, pd.DataFrame)

    def test_str(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        self.assertIn("Class BoutsNLS object", xbouts.__str__())

    def test_init_pars(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)

        init_pars = xbouts.init_pars([80], plot=False)
        self.assertIsInstance(init_pars, pd.DataFrame)
        self.assertEqual(init_pars.size, 4)

        fig, ax = plt.subplots()
        _ = xbouts.init_pars([80], plot=True, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 3)
        plt.close()

    def test_fit(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        self.assertIsInstance(coefs, pd.DataFrame)
        self.assertEqual(coefs.size, 4)
        self.assertEqual(pcov.size, 16)

    def test_bec(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        self.assertIsInstance(bec, np.ndarray)
        self.assertEqual(bec.size, 1)

    def test_label_bouts(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)
        bec = xbouts.bec(coefs)
        xlabeled = skbouts.label_bouts(x, bec)
        self.assertIsInstance(xlabeled, type(x))
        self.assertEqual(xlabeled.shape, x.shape)

    def test_plot_fit(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_fit(coefs, ax=ax)
        line = ax.get_lines()
        self.assertEqual(len(line), 1)
        plt.close()

    def test_plot_ecdf(self):
        x = self.x
        xbouts = skbouts.BoutsNLS(x, 5)
        init_pars = xbouts.init_pars([80], plot=False)

        coefs, pcov = xbouts.fit(init_pars)

        fig, ax = plt.subplots()
        _ = xbouts.plot_ecdf(coefs, ax=ax)
        lines = ax.get_lines()
        self.assertEqual(len(lines), 2)
        plt.close()


if __name__ == '__main__':
    ut.main()
