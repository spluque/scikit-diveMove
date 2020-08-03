"""BoutsMLE class

"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logit, expit
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from . import bouts

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


def mle_fun(x, p, lambdas):
    r"""Random Poisson processes function

    The current implementation takes two or three random Poisson processes.

    Parameters
    ----------
    x : array_like
        Independent data array described by model with parameters `p`,
        :math:`\lambda_f`, and :math:`\lambda_s`.
    p : list
        List with mixing parameters of the model.
    lambdas : array_like
        1-D Array with the density parameters (:math:`\lambda`) of the
        model.  Its length must be length(p) + 1.

    Returns
    -------
    out : array_like
        Same shape as `x` with the evaluated function.

    """
    logmsg = "p={0}, lambdas={1}".format(p, lambdas)
    logger.info(logmsg)
    ncoefs = lambdas.size

    # We assume at least two processes
    p0 = p[0]
    lda0 = lambdas[0]
    term0 = p0 * lda0 * np.exp(-lda0 * x)

    if ncoefs == 2:
        lda1 = lambdas[1]
        term1 = (1 - p0) * lda1 * np.exp(-lda1 * x)
        res = term0 + term1
    elif ncoefs == 3:
        p1 = p[1]
        lda1 = lambdas[1]
        term1 = p1 * (1 - p0) * lda1 * np.exp(-lda1 * x)
        lda2 = lambdas[2]
        term2 = (1 - p1) * (1 - p0) * lda2 * np.exp(-lda2 * x)
        res = term0 + term1 + term2
    else:
        msg = "Only mixtures of <= 3 processes are implemented"
        raise KeyError(msg)

    return(np.log(res))


class BoutsMLE(bouts.Bouts):
    """Nonlinear least squares bout identification

    """

    def loglik_fun(self, params, x, transformed=True):
        r"""Log likelihood function of parameters given observed data

        Parameters
        ----------
        params : array_like
            1-D array with parameters to fit.  Currently must be 3-length,
            with mixing parameter :math:`p`, density parameter
            :math:`\lambda_f` and :math:`\lambda_s`, in that order.
        x : array_like
            Independent data array described by model with parameters `p`,
            :math:`\lambda_f`, and :math:`\lambda_s`.
        transformed : bool
            Whether `params` are transformed and need to be un-transformed
            to calculate the likelihood.

        Returns
        -------
        out :

        """
        p = params[0]
        lambdas = params[1:]

        if transformed:
            p = expit(p)
            lambdas = np.exp(lambdas)

        # Need list `p` for mle_fun
        ll = -sum(mle_fun(x, [p], lambdas))
        logger.info("LL={}".format(ll))
        return(ll)

    def fit(self, start, fit1_opts=None, fit2_opts=None):
        """Maximum likelihood estimation of log frequencies

        Parameters
        ----------
        start : pandas.DataFrame
            DataFrame with starting values for coefficients of each process
            in columns.  These can come from the "broken stick" method as
            in :meth:`Bouts.init_pars`, and will be transformed to minimize
            the first log likelihood function.
        fit1_opts, fit2_opts : dict
            Dictionaries with keywords to be pass to
            :func:`scipy.optimize.minimize`, for the first and second fits.

        Returns
        -------
        fit1, fit2 : scipy.optimize.OptimizeResult
            Objects with the optimization result from the first and second
            fit, having a `x` attribute with coefficients of the solution.

        Notes
        -----
        Current implementation handles mixtures of two Poisson processes.

        """
        # Calculate `p`
        p0, lambda0 = bouts.calc_p(start)
        # transform parameters for first fit
        lambda0 = np.log(lambda0)
        x0 = np.array([*logit(p0), *lambda0])

        if fit1_opts:
            fit1 = minimize(self.loglik_fun, x0=x0, args=(self.x,),
                            **fit1_opts)
        else:
            fit1 = minimize(self.loglik_fun, x0=x0, args=(self.x,))

        coef0 = fit1.x

        start2 = [expit(coef0[0]), *np.exp(coef0[1:])]
        if fit2_opts:
            fit2 = minimize(self.loglik_fun, x0=start2,
                            args=(self.x, False), **fit2_opts)
        else:
            fit2 = minimize(self.loglik_fun, x0=start2,
                            args=(self.x, False))
        logger.info("N iter fit 1: {0}, fit 2: {1}"
                    .format(fit1.nit, fit2.nit))

        return(fit1, fit2)

    def bec(self, fit):
        """Calculate bout ending criteria from model coefficients

        Parameters
        ----------
        fit : scipy.optimize.OptimizeResult
            Object with the optimization result, having a `x` attribute
            with coefficients of the solution.

        Returns
        -------
        out : ndarray

        Notes
        -----
        Current implementation is for a two-process mixture, hence an array
        of a single float is returned.

        """
        coefs = fit.x

        p_hat = coefs[0]
        lambda1_hat = coefs[1]
        lambda2_hat = coefs[2]
        bec = (np.log((p_hat * lambda1_hat) / ((1 - p_hat) * lambda2_hat)) /
               (lambda1_hat - lambda2_hat))

        return(np.array(bec))

    def plot_fit(self, fit, ax=None):
        """Plot log frequency histogram and fitted model

        Parameters
        ----------
        fit : scipy.optimize.OptimizeResult
            Object with the optimization result, having a `x` attribute
            with coefficients of the solution.
        ax : matplotlib.Axes instance
            An Axes instance to use as target.

        Returns
        -------
        ax : `matplotlib.Axes`

        """
        # Method is redefined from Bouts
        x = self.x
        coefs = fit.x
        p_hat = coefs[0]
        lambdas_hat = coefs[1:]
        xmin = x.min()
        xmax = x.max()
        # BEC
        becx = self.bec(fit)
        becy = mle_fun(becx, [p_hat], lambdas_hat)

        x_pred = np.linspace(xmin, xmax, num=101)  # matches R's curve
        # Need to transpose to unpack columns rather than rows
        y_pred = mle_fun(x_pred, [p_hat], lambdas_hat)

        if ax is None:
            ax = plt.gca()

        # Data rug plot
        ax.plot(x, np.ones_like(x) * y_pred.max(), "|",
                color="k", label="observed")
        # Plot predicted
        ax.plot(x_pred, y_pred, label="model")
        # Plot BEC
        ylim = ax.get_ylim()
        ax.vlines(becx, ylim[0], becy, linestyle="--")

        # Annotations
        ax.annotate("bec = {0:.3f}".format(becx), (becx, ylim[0]),
                    xytext=(5, 0), textcoords="offset points")

        ax.legend(loc=8, bbox_to_anchor=(0.5, 1), frameon=False,
                  borderaxespad=0.1, ncol=2)
        ax.set_xlabel("x")
        ax.set_ylabel("log frequency")

        return(ax)

    def plot_ecdf(self, fit, ax=None):
        """Plot observed and modelled empirical cumulative frequencies

        Parameters
        ----------
        fit : scipy.optimize.OptimizeResult
            Object with the optimization result, having a `x` attribute
            with coefficients of the solution.
        ax : matplotlib.Axes instance
            An Axes instance to use as target.

        Returns
        -------
        ax : `matplotlib.Axes`

        """
        x = self.x

        coefs = fit.x
        xx = np.log1p(x)
        x_ecdf = ECDF(xx)
        x_pred = np.linspace(0, xx.max(), num=101)
        y_pred = x_ecdf(x_pred)

        if ax is None:
            ax = plt.gca()

        # Plot ECDF of data
        ax.step(np.expm1(x_pred), y_pred, label="observed")
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(np.exp(xx).min(), np.exp(xx).max())
        # Plot estimated CDF
        p = [coefs[0]]          # list to bouts.ecdf()
        lambdas = pd.Series([coefs[1], coefs[2]], name="lambda")
        y_mod = bouts.ecdf(np.expm1(x_pred), p, lambdas)
        ax.plot(np.expm1(x_pred), y_mod, label="model")
        # Add a little offset to ylim for visibility
        yoffset = (0.05, 1.05)
        ax.set_ylim(*yoffset)       # add some spacing
        # Plot BEC
        becx = self.bec(fit)
        becy = bouts.ecdf(becx, p, lambdas)
        ax.vlines(becx, 0, becy, linestyle="--")
        # Annotations
        ax.legend(loc="upper left")
        ax.annotate("bec = {0:.3f}".format(becx),
                    (becx, yoffset[0]), xytext=(5, 5),
                    textcoords="offset points")
        ax.set_xlabel("x")
        ax.set_ylabel("ECDF [x]")

        return(ax)


if __name__ == '__main__':
    # Set up info level logging
    logging.basicConfig(level=logging.INFO)
    from skdiveMove.tests import diveMove2skd

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
    bouts_postdive = BoutsMLE(postdives_diff, 0.1)
    # Get init parameters from broken stick model
    bout_init_pars = bouts_postdive.init_pars([50], plot=False)

    # Knowing
    p_bnd = (-2, None)
    lda1_bnd = (-5, None)
    lda2_bnd = (-10, None)
    bd1 = (p_bnd, lda1_bnd, lda2_bnd)
    p_bnd = (1e-8, None)
    lda1_bnd = (1e-8, None)
    lda2_bnd = (1e-8, None)
    bd2 = (p_bnd, lda1_bnd, lda2_bnd)
    fit1, fit2 = bouts_postdive.fit(bout_init_pars,
                                    fit1_opts=dict(method="L-BFGS-B",
                                                   bounds=bd1),
                                    fit2_opts=dict(method="L-BFGS-B",
                                                   bounds=bd2))

    # BEC
    becx = bouts_postdive.bec(fit2)
    ax = bouts_postdive.plot_fit(fit2)
    bouts_postdive.plot_ecdf(fit2)
