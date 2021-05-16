"""BoutsNLS class

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from statsmodels.distributions.empirical_distribution import ECDF
from . import bouts


class BoutsNLS(bouts.Bouts):
    """Nonlinear Least Squares fitting for models of Poisson process mixtures

    Methods for modelling log-frequency data as a mixture of Poisson
    processes via nonlinear least squares [1]_.

    References
    ----------
    .. [1] Sibly, R.; Nott, H. and Fletcher, D. (1990) Splitting behaviour
       into bouts Animal Behaviour 39, 63-69.

    Examples
    --------
    Draw 1000 samples from a mixture where the first process occurs with
    :math:`p < 0.7` and the second process occurs with the remaining
    probability.

    >>> from skdiveMove.tests import random_mixexp
    >>> rng = np.random.default_rng(123)
    >>> x2 = random_mixexp(1000, p=0.7, lda=np.array([0.05, 0.005]),
    ...                    rng=rng)
    >>> xbouts2 = BoutsNLS(x2, bw=5)
    >>> init_pars = xbouts2.init_pars([80], plot=False)

    Fit the model and retrieve coefficients:

    >>> coefs, pcov = xbouts2.fit(init_pars)
    >>> print(np.round(coefs, 4))
            (2.519, 80.0]  (80.0, 1297.52]
    a           3648.8547        1103.4423
    lambda         0.0388           0.0032

    Calculate bout-ending criterion (returns array):

    >>> print(np.round(xbouts2.bec(coefs), 4))
    [103.8648]

    Plot observed and predicted data:

    >>> xbouts2.plot_fit(coefs)  # doctest: +ELLIPSIS
    <AxesSubplot:...>

    Plot ECDF:

    >>> xbouts2.plot_ecdf(coefs)  # doctest: +ELLIPSIS
    <AxesSubplot:...>

    """
    def fit(self, start, **kwargs):
        """Fit non-linear least squares model to log frequencies

        The metaclass :class:`bouts.Bouts` implements this method.

        Parameters
        ----------
        start : pandas.DataFrame
            DataFrame with coefficients for each process in columns.
        **kwargs : optional keyword arguments
            Passed to `scipy.optimize.curve_fit`.

        Returns
        -------
        coefs : pandas.DataFrame
            Coefficients of the model.
        pcov : 2D array
            Covariance of coefs.

        """
        return(bouts.Bouts.fit(self, start, **kwargs))

    def bec(self, coefs):
        """Calculate bout ending criteria from model coefficients

        The metaclass :class:`bouts.Bouts` implements this method.

        Parameters
        ----------
        coefs : pandas.DataFrame
            DataFrame with model coefficients in columns.

        Returns
        -------
        out : ndarray, shape (n,)
            1-D array with BECs implied by `coefs`.  Length is
            coefs.shape[1]

        """
        # The metaclass implements this method
        return(bouts.Bouts.bec(self, coefs))

    def plot_ecdf(self, coefs, ax=None, **kwargs):
        """Plot observed and modelled empirical cumulative frequencies

        Parameters
        ----------
        coefs : pandas.DataFrame
            DataFrame with model coefficients in columns.
        ax : matplotlib.Axes instance
            An Axes instance to use as target.
        **kwargs : optional keyword arguments
            Passed to `matplotlib.pyplot.gca`.

        Returns
        -------
        ax : `matplotlib.Axes`

        """
        x = self.x

        xx = np.log1p(x)
        x_ecdf = ECDF(xx)
        x_pred = np.linspace(0, xx.max(), num=101)
        x_pred_expm1 = np.expm1(x_pred)
        y_pred = x_ecdf(x_pred)

        if ax is None:
            ax = plt.gca(**kwargs)

        # Plot ECDF of data
        ax.step(x_pred_expm1, y_pred, label="observed")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(np.exp(xx).min(), np.exp(xx).max())
        # Plot estimated CDF
        p, lambdas = bouts.calc_p(coefs)
        y_mod = bouts.ecdf(x_pred_expm1, p, lambdas)
        ax.plot(x_pred_expm1, y_mod, label="model")
        # Add a little offset to ylim for visibility
        yoffset = (0.05, 1.05)
        ax.set_ylim(*yoffset)       # add some spacing
        # Plot BEC
        bec_x = self.bec(coefs)
        bec_y = bouts.ecdf(bec_x, p=p, lambdas=lambdas)
        bouts._plot_bec(bec_x, bec_y=bec_y, ax=ax, xytext=(-5, 5),
                        horizontalalignment="right")
        ax.legend(loc="upper left")

        ax.set_xlabel("x")
        ax.set_ylabel("ECDF [x]")

        return(ax)


if __name__ == '__main__':
    from skdiveMove.tests import diveMove2skd
    import pandas as pd

    tdrX = diveMove2skd()
    pars = {"offset_zoc": 3,
            "dry_thr": 70,
            "wet_thr": 3610,
            "dive_thr": 3,
            "dive_model": "unimodal",
            "smooth_par": 0.1,
            "knot_factor": 20,
            "descent_crit_q": 0.01,
            "ascent_crit_q": 0}

    tdrX.calibrate(zoc_method="offset", offset=pars["offset_zoc"],
                   dry_thr=pars["dry_thr"], wet_thr=pars["dry_thr"],
                   dive_thr=pars["dive_thr"],
                   dive_model=pars["dive_model"],
                   smooth_par=pars["smooth_par"],
                   knot_factor=pars["knot_factor"],
                   descent_crit_q=pars["descent_crit_q"],
                   ascent_crit_q=pars["ascent_crit_q"])
    stats = tdrX.dive_stats()
    stamps = tdrX.stamp_dives(ignore_z=True)
    stats_tab = pd.concat((stamps, stats), axis=1)
    # 2=4 here
    postdives = stats_tab["postdive_dur"][stats_tab["phase_id"] == 4]
    postdives_diff = postdives.dt.total_seconds().diff()[1:].abs()
    # Remove isolated dives
    postdives_diff = postdives_diff[postdives_diff < 2000]

    # Set up instance
    bouts_postdive = BoutsNLS(postdives_diff, 0.1)
    # Get init parameters
    bout_init_pars = bouts_postdive.init_pars([50], plot=False)
    nls_coefs, _ = bouts_postdive.fit(bout_init_pars)
    # BEC
    bouts_postdive.bec(nls_coefs)
    bouts_postdive.plot_fit(nls_coefs)
    # ECDF
    fig1, ax1 = bouts_postdive.plot_ecdf(nls_coefs)
    # Try 3 processes
    # Get init parameters
    bout_init_pars = bouts_postdive.init_pars([50, 550], plot=False)
    nls_coefs, _ = bouts_postdive.fit(bout_init_pars)
    # BEC
    bouts_postdive.bec(nls_coefs)
    bouts_postdive.plot_fit(nls_coefs)
    # ECDF
    fig2, ax2 = bouts_postdive.plot_ecdf(nls_coefs)
