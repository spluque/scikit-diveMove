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


def mleLL(x, p, lambdas):
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
    nproc = lambdas.size

    # We assume at least two processes
    p0 = p[0]
    lda0 = lambdas[0]
    term0 = p0 * lda0 * np.exp(-lda0 * x)

    if nproc == 2:
        lda1 = lambdas[1]
        term1 = (1 - p0) * lda1 * np.exp(-lda1 * x)
        res = term0 + term1
    else:                # 3 processes; capabilities enforced in mleLL
        p1 = p[1]
        lda1 = lambdas[1]
        term1 = p1 * (1 - p0) * lda1 * np.exp(-lda1 * x)
        lda2 = lambdas[2]
        term2 = (1 - p1) * (1 - p0) * lda2 * np.exp(-lda2 * x)
        res = term0 + term1 + term2

    return(np.log(res))


class BoutsMLE(bouts.Bouts):
    r"""Maximum Likelihood estimation for models of Poisson process mixtures

    Methods for modelling log-frequency data as a mixture of Poisson
    processes via maximum likelihood estimation [2]_, [3]_.  Mixtures of
    two or three Poisson processes are supported.

    Even in these relatively simple cases, it is very important to provide
    good starting values for the parameters.

    One useful strategy to get good starting parameter values is to proceed
    in 4 steps.  First, fit a broken stick model to the log frequencies of
    binned data (see :meth:`~Bouts.init_pars`), to obtain estimates of 4
    parameters in a 2-process model [1]_, or 6 in a 3-process model.
    Second, calculate parameter(s) :math:`p` from the :math:`\alpha`
    parameters obtained by fitting the broken stick model, to get tentative
    initial values as in [2]_.  Third, obtain MLE estimates for these
    parameters, but using a reparameterized version of the -log L2
    function.  Lastly, obtain the final MLE estimates for the three
    parameters by using the estimates from step 3, un-transformed back to
    their original scales, maximizing the original parameterization of the
    -log L2 function.

    :meth:`~Bouts.init_pars` can be used to perform step 1.  Calculation of
    the mixing parameters :math:`p` in step 2 is trivial from these
    estimates.  Method :meth:`negMLEll` calculates the negative
    log-likelihood for a reparameterized version of the -log L2 function
    given by [1]_, so can be used for step 3.  This uses a logit
    transformation of the mixing parameter :math:`p`, and log
    transformations for density parameters :math:`\lambda`.  Method
    :meth:`negMLEll` is used again to compute the -log L2 function
    corresponding to the un-transformed model for step 4.

    The :meth:`fit` method performs the main job of maximizing the -log L2
    functions, and is essentially a wrapper around
    :func:`~scipy.optimize.minimize`.  It only takes the -log L2 function,
    a `DataFrame` of starting values, and the variable to be modelled, all
    of which are passed to :func:`~scipy.optimize.minimize` for
    optimization.  Additionally, any other arguments are also passed to
    :func:`~scipy.optimize.minimize`, hence great control is provided for
    fitting any of the -log L2 functions.

    In practice, step 3 does not pose major problems using the
    reparameterized -log L2 function, but it might be useful to use method
    'L-BFGS-B' with appropriate lower and upper bounds.  Step 4 can be a
    bit more problematic, because the parameters are usually on very
    different scales and there can be multiple minima.  Therefore, it is
    almost always the rule to use method 'L-BFGS-B', again bounding the
    parameter search, as well as other settings for controlling the
    optimization.

    References
    ----------
    .. [2] Langton, S.; Collett, D. and Sibly, R. (1995) Splitting
       behaviour into bouts; a maximum likelihood approach.  Behaviour 132,
       9-10.

    .. [3] Luque, S.P. and Guinet, C. (2007) A maximum likelihood approach
       for identifying dive bouts improves accuracy, precision, and
       objectivity. Behaviour, 144, 1315-1332.

    Examples
    --------
    See :doc:`boutsimuldemo` for a detailed example.

    """

    def negMLEll(self, params, x, istransformed=True):
        r"""Log likelihood function of parameters given observed data

        Parameters
        ----------
        params : array_like
            1-D array with parameters to fit.  Currently must be either
            3-length, with mixing parameter :math:`p`, density parameter
            :math:`\lambda_f` and :math:`\lambda_s`, in that order, or
            5-length, with :math:`p_f`, :math:`p_fs`, :math:`\lambda_f`,
            :math:`\lambda_m`, :math:`\lambda_s`.
        x : array_like
            Independent data array described by model with parameters
            `params`.
        istransformed : bool
            Whether `params` are transformed and need to be un-transformed
            to calculate the likelihood.

        Returns
        -------
        out :

        """
        if len(params) == 3:
            # Need list `p` for mle_fun
            p = [params[0]]
            lambdas = params[1:]
        elif len(params) == 5:
            p = params[:2]
            lambdas = params[2:]
        else:
            msg = "Only mixtures of <= 3 processes are implemented"
            raise KeyError(msg)

        if istransformed:
            p = expit(p)
            lambdas = np.exp(lambdas)

        ll = -sum(mleLL(x, p, lambdas))
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
            Dictionaries with keywords to be passed to
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

        logger.info("Starting first fit")
        if fit1_opts:
            fit1 = minimize(self.negMLEll, x0=x0, args=(self.x,),
                            **fit1_opts)
        else:
            fit1 = minimize(self.negMLEll, x0=x0, args=(self.x,))

        coef0 = fit1.x

        start2 = [expit(coef0[0]), *np.exp(coef0[1:])]
        logger.info("Starting second fit")
        if fit2_opts:
            fit2 = minimize(self.negMLEll, x0=start2,
                            args=(self.x, False), **fit2_opts)
        else:
            fit2 = minimize(self.negMLEll, x0=start2,
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

        if len(coefs) == 3:
            p_hat = coefs[0]
            lda1_hat = coefs[1]
            lda2_hat = coefs[2]
            bec = (np.log((p_hat * lda1_hat) /
                          ((1 - p_hat) * lda2_hat)) /
                   (lda1_hat - lda2_hat))
        elif len(coefs) == 5:
            p0_hat, p1_hat = coefs[:2]
            lda0_hat, lda1_hat, lda2_hat = coefs[2:]
            bec0 = (np.log((p0_hat * lda0_hat) /
                           ((1 - p0_hat) * lda1_hat)) /
                    (lda0_hat - lda1_hat))
            bec1 = (np.log((p1_hat * lda1_hat) /
                           ((1 - p1_hat) * lda2_hat)) /
                    (lda1_hat - lda2_hat))
            bec = [bec0, bec1]

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
        if len(coefs) == 3:
            p_hat = [coefs[0]]
            lda_hat = coefs[1:]
        elif len(coefs) == 5:
            p_hat = coefs[:2]
            lda_hat = coefs[2:]
        xmin = x.min()
        xmax = x.max()

        x_pred = np.linspace(xmin, xmax, num=101)  # matches R's curve
        # Need to transpose to unpack columns rather than rows
        y_pred = mleLL(x_pred, p_hat, lda_hat)

        if ax is None:
            ax = plt.gca()

        # Data rug plot
        ax.plot(x, np.ones_like(x) * y_pred.max(), "|",
                color="k", label="observed")
        # Plot predicted
        ax.plot(x_pred, y_pred, label="model")
        # Plot BEC
        bec_x = self.bec(fit)
        bec_y = mleLL(bec_x, p_hat, lda_hat)
        bouts._plot_bec(bec_x, bec_y, ax=ax, xytext=(5, 5))

        ax.legend(loc=8, bbox_to_anchor=(0.5, 1), frameon=False,
                  borderaxespad=0.1, ncol=2)
        ax.set_xlabel("x")
        ax.set_ylabel("log frequency")

        return(ax)

    def plot_ecdf(self, fit, ax=None, **kwargs):
        """Plot observed and modelled empirical cumulative frequencies

        Parameters
        ----------
        fit : scipy.optimize.OptimizeResult
            Object with the optimization result, having a `x` attribute
            with coefficients of the solution.
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
        coefs = fit.x
        if len(coefs) == 3:
            p_hat = [coefs[0]]          # list to bouts.ecdf()
            lda_hat = pd.Series(coefs[1:], name="lambda")
        elif len(coefs) == 5:
            p_hat = coefs[:2]
            lda_hat = pd.Series(coefs[2:], name="lambda")
        y_mod = bouts.ecdf(x_pred_expm1, p_hat, lda_hat)
        ax.plot(x_pred_expm1, y_mod, label="model")
        # Add a little offset to ylim for visibility
        yoffset = (0.05, 1.05)
        ax.set_ylim(*yoffset)       # add some spacing
        # Plot BEC
        bec_x = self.bec(fit)
        bec_y = bouts.ecdf(bec_x, p=p_hat, lambdas=lda_hat)
        bouts._plot_bec(bec_x, bec_y=bec_y, ax=ax, xytext=(-5, 5),
                        horizontalalignment="right")
        ax.legend(loc="upper left")

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
