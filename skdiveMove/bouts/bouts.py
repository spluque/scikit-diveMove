"""Abstract class `Bouts` for Poisson mixture models

This module also provides useful functions for other modules subclassing
:class:`Bouts`.

"""

import logging
from abc import ABCMeta, abstractmethod
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from skdiveMove.helpers import rle_key

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


def nlsLL(x, coefs):
    r"""Generalized log-likelihood for Random Poisson mixtures

    This is a generalized form taking any number of Poisson processes.

    Parameters
    ----------
    x : array_like
        Independent data array described by the function
    coefs : array_like
        2-D array with coefficients ('a', :math:'\lambda') in rows for each
        process of the model in columns.

    Returns
    -------
    out : array_like
        Same shape as `x` with the evaluated log-likelihood.

    """
    def calc_term(params):
        return(params[0] * params[1] * np.exp(-params[1] * x))

    terms = np.apply_along_axis(calc_term, 0, coefs)
    terms_sum = terms.sum(1)
    if np.any(terms_sum <= 0):
        logger.warning("Negative values at: {}".format(coefs))
    return(np.log(terms_sum))


def calc_p(coefs):
    r"""Calculate `p` (proportion) parameter from `a` coefficients

    Parameters
    ----------
    coefs : pandas.DataFrame
        DataFrame with model coefficients in columns, and indexed by
        parameter names "a" and "lambda".

    Returns
    -------
    p : list
        Proportion parameters implied in `coef`.
    lambdas : pandas.Series
        A series with with the :math:`\lambda` parameters from `coef`.

    """
    ncoefs = coefs.shape[1]
    coef_arr = np.arange(ncoefs)
    pairs = [(i, i + 1) for i in coef_arr[:-1]]
    p_ll = []               # build mixing ratios

    for pair in pairs:
        procn1 = coefs.columns[pair[0]]  # name of process 1
        procn2 = coefs.columns[pair[1]]  # name of process 2
        a1 = coefs.loc["a", procn1]
        a2 = coefs.loc["a", procn2]
        p_i = a1 / (a1 + a2)
        p_ll.append(p_i)

    return(p_ll, coefs.loc["lambda"])


def ecdf(x, p, lambdas):
    r"""Estimated cumulative frequency for Poisson mixture models

    ECDF for two- or three-process mixture models.

    Parameters
    ----------
    x : array_like
        Independent data array described by model with parameters `p`,
        :math:`\lambda_f`, and :math:`\lambda_s`.
    p : list
        List with mixing parameters of the model.
    lambdas : pandas.Series
        Series with the density parameters (:math:`\lambda`) of the
        model.  Its length must be length(p) + 1.

    Returns
    -------
    out : array_like
        Same shape as `x` with the evaluated function.

    """
    ncoefs = lambdas.size

    # We assume at least two processes
    p0 = p[0]
    lda0 = lambdas.iloc[0]
    term0 = 1 - p0 * np.exp(-lda0 * x)

    if ncoefs == 2:
        lda1 = lambdas.iloc[1]
        term1 = (1 - p0) * np.exp(-lda1 * x)
        cdf = term0 - term1
    elif ncoefs == 3:
        p1 = p[1]
        lda1 = lambdas.iloc[1]
        term1 = p1 * (1 - p0) * np.exp(-lda1 * x)
        lda2 = lambdas.iloc[2]
        term2 = (1 - p0) * (1 - p1) * np.exp(-lda2 * x)
        cdf = term0 - term1 - term2
    else:
        msg = "Only mixtures of <= 3 processes are implemented"
        raise KeyError(msg)

    return(cdf)


def label_bouts(x, bec, as_diff=False):
    """Classify data into bouts based on bout ending criteria

    Parameters
    ----------
    x : pandas.Series
        Series with data to classify according to `bec`.
    bec : array_like
        Array with bout-ending criteria. It is assumed to be sorted.
    as_diff : bool, optional
        Whether to apply `diff` on `x` so it matches `bec`'s scale.

    Returns
    -------
    out : ndarray
        Integer array with the same shape as `x`.

    """
    if as_diff:
        xx = x.diff().fillna(0)

    else:
        xx = x.copy()

    xx_min = np.array(xx.min())
    xx_max = np.array(xx.max())
    brks = np.append(np.append(xx_min, bec), xx_max)
    xx_cat = pd.cut(xx, bins=brks, include_lowest=True)
    xx_bouts = rle_key(xx_cat)

    return(xx_bouts)


def _plot_bec(bec_x, bec_y, ax, xytext, horizontalalignment="left"):
    """Plot bout-ending criteria on `Axes`

    Private helper function only for convenience here.

    Parameters
    ----------
    bec_x : ndarray, shape (n,)
        x coordinate for bout-ending criteria.
    bec_y : ndarray, shape (n,)
        y coordinate for bout-ending criteria.
    ax : matplotlib.Axes
        An Axes instance to use as target.
    xytext : 2-tuple
        Argument passed to `matplotlib.annotate`; interpreted with
        textcoords="offset points".
    horizontalalignment : str
        Argument passed to `matplotlib.annotate`.

    """
    ylims = ax.get_ylim()
    ax.vlines(bec_x, ylims[0], bec_y, linestyle="--")
    ax.scatter(bec_x, bec_y, c="r", marker="v")
    # Annotations
    fmtstr = "bec_{0} = {1:.3f}"
    if bec_x.size == 1:
        bec_x = bec_x.item()
        ax.annotate(fmtstr.format(0, bec_x),
                    (bec_x, bec_y), xytext=xytext,
                    textcoords="offset points",
                    horizontalalignment=horizontalalignment)
    else:
        for i, bec_i in enumerate(bec_x):
            ax.annotate(fmtstr.format(i, bec_i),
                        (bec_i, bec_y[i]), xytext=xytext,
                        textcoords="offset points",
                        horizontalalignment=horizontalalignment)


class Bouts(metaclass=ABCMeta):
    """Abstract base class for models of log-transformed frequencies

    This is a base class for other classes to build on, and do the model
    fitting.  `Bouts` is an abstract base class to set up bout
    identification procedures.  Subclasses must implement `fit` and `bec`
    methods, or re-use the default NLS methods in `Bouts`.

    Attributes
    ----------
    x : array_like
        1D array with input data.
    method : str
        Method used for calculating the histogram.
    lnfreq : pandas.DataFrame
        DataFrame with the centers of histogram bins, and corresponding
        log-frequencies of `x`.

    """
    def __init__(self, x, bw, method="standard"):
        """Histogram of log transformed frequencies of `x`

        Parameters
        ----------
        x : array_like
            1D array with data where bouts will be identified based on
            `method`.
        bw : float
            Bin width for the histogram
        method : {"standard", "seq_diff"}, optional
            Method to use for calculating the frequencies: "standard"
            simply uses `x`, which "seq_diff" uses the sequential
            differences method.
        **kwargs : optional keywords
            Passed to histogram

        """
        self.x = x
        self.method = method
        if method == "standard":
            upper = x.max()
            brks = np.arange(x.min(), upper, bw)
            if brks[-1] < upper:
                brks = np.append(brks, brks[-1] + bw)
            h, edges = np.histogram(x, bins=brks)
        elif method == "seq_diff":
            x_diff = np.abs(np.diff(x))
            upper = x_diff.max()
            brks = np.arange(0, upper, bw)
            if brks[-1] < upper:
                brks = np.append(brks, brks[-1] + bw)
            h, edges = np.histogram(x_diff, bins=brks)

        ctrs = edges[:-1] + np.diff(edges) / 2
        ok = h > 0
        ok_at = np.where(ok)[0] + 1  # 1-based indices
        freq_adj = h[ok] / np.diff(np.insert(ok_at, 0, 0))

        self.lnfreq = pd.DataFrame({"x": ctrs[ok],
                                    "lnfreq": np.log(freq_adj)})

    def __str__(self):
        method = self.method
        lnfreq = self.lnfreq
        objcls = ("Class {} object\n".format(self.__class__.__name__))
        meth_str = "{0:<20} {1}\n".format("histogram method: ", method)
        lnfreq_str = ("{0:<20}\n{1}"
                      .format("log-frequency histogram:",
                              lnfreq.describe()))
        return(objcls + meth_str + lnfreq_str)

    def init_pars(self, x_break, plot=True, ax=None, **kwargs):
        """Find starting values for mixtures of random Poisson processes

        Starting values are calculated using the "broken stick" method.

        Parameters
        ----------
        x_break : array_like
            One- or two-element array with values determining the break(s)
            for broken stick model, such that x < x_break[0] is first
            process, x >= x_break[1] & x < x_break[2] is second process,
            and x >= x_break[2] is third one.
        plot : bool, optional
            Whether to plot the broken stick model.
        ax : matplotlib.Axes, optional
            An Axes instance to use as target.  Default is to create one.
        **kwargs : optional keyword arguments
            Passed to plotting function.

        Returns
        -------
        out : pandas.DataFrame
            DataFrame with coefficients for each process.

        """
        nproc = len(x_break)
        if (nproc > 2):
            msg = "x_break must be length <= 2"
            raise IndexError(msg)

        lnfreq = self.lnfreq
        ctrs = lnfreq["x"]
        xmin = ctrs.min()
        xmax = ctrs.max()
        xbins = [xmin]
        xbins.extend(x_break)
        xbins.extend([xmax])
        procf = pd.cut(ctrs, bins=xbins, right=True,
                       include_lowest=True)
        lnfreq_grp = lnfreq.groupby(procf)

        coefs_ll = []
        for name, grp in lnfreq_grp:
            fit = smf.ols("lnfreq ~ x", data=grp).fit()
            coefs_ll.append(fit.params.rename(name))

        coefs = pd.concat(coefs_ll, axis=1)

        def calculate_pars(p):
            """Poisson process parameters from linear model

            """
            lda = -p["x"]
            a = np.exp(p["Intercept"]) / lda
            return(pd.Series({"a": a, "lambda": lda}))

        pars = coefs.apply(calculate_pars)

        if plot:

            if ax is None:
                ax = plt.gca()

            freq_min = lnfreq["lnfreq"].min()
            freq_max = lnfreq["lnfreq"].max()
            for name, grp in lnfreq_grp:
                ax.scatter(x="x", y="lnfreq", data=grp, label=name)
                # Plot current "stick"
                coef_i = coefs[name]
                y_stick = coef_i["Intercept"] + ctrs * coef_i["x"]
                # Limit the "stick" line to min/max of data
                ok = (y_stick >= freq_min) & (y_stick <= freq_max)
                ax.plot(ctrs[ok], y_stick[ok], linestyle="--")

            x_pred = np.linspace(xmin, xmax, num=101)  # matches R's curve
            y_pred = nlsLL(x_pred, pars)
            ax.plot(x_pred, y_pred, alpha=0.5, label="model")
            ax.legend(loc="upper right")
            ax.set_xlabel("x")
            ax.set_ylabel("log frequency")

        return(pars)

    @abstractmethod
    def fit(self, start, **kwargs):
        """Fit Poisson mixture model to log frequencies

        Default is non-linear least squares method.

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
        lnfreq = self.lnfreq
        xdata = lnfreq["x"]
        ydata = lnfreq["lnfreq"]

        def _nlsLL(x, *args):
            """Wrapper to nlsLL to allow for array argument"""
            # Pass in original shape, damn it!  Note order="F" needed
            coefs = np.array(args).reshape(start.shape, order="F")
            return(nlsLL(x, coefs))

        # Rearrange starting values into a 1D array (needs to be flat)
        init_flat = start.to_numpy().T.reshape((start.size,))
        popt, pcov = curve_fit(_nlsLL, xdata, ydata,
                               p0=init_flat, **kwargs)
        # Reshape coefs back into init shape
        coefs = pd.DataFrame(popt.reshape(start.shape, order="F"),
                             columns=start.columns, index=start.index)
        return(coefs, pcov)

    @abstractmethod
    def bec(self, coefs):
        """Calculate bout ending criteria from model coefficients

        Implementing default as from NLS method.

        Parameters
        ----------
        coefs : pandas.DataFrame
            DataFrame with model coefficients in columns, and indexed by
            parameter names "a" and "lambda".

        Returns
        -------
        out : ndarray, shape (n,)
            1-D array with BECs implied by `coefs`.  Length is
            coefs.shape[1]

        """
        # Find bec's per process by pairing columns
        ncoefs = coefs.shape[1]
        coef_arr = np.arange(ncoefs)
        pairs = [(i, i + 1) for i in coef_arr[:-1]]
        becs = []
        for pair in pairs:
            procn1 = coefs.columns[pair[0]]  # name of process 1
            procn2 = coefs.columns[pair[1]]  # name of process 2
            a1 = coefs.loc["a", procn1]
            lambda1 = coefs.loc["lambda", procn1]
            a2 = coefs.loc["a", procn2]
            lambda2 = coefs.loc["lambda", procn2]
            bec = (np.log((a1 * lambda1) / (a2 * lambda2)) /
                   (lambda1 - lambda2))
            becs.append(bec)

        return(np.array(becs))

    def plot_fit(self, coefs, ax=None):
        """Plot log frequency histogram and fitted model

        Parameters
        ----------
        coefs : pandas.DataFrame
            DataFrame with model coefficients in columns, and indexed by
            parameter names "a" and "lambda".
        ax : matplotlib.Axes instance
            An Axes instance to use as target.

        Returns
        -------
        ax : `matplotlib.Axes`

        """
        lnfreq = self.lnfreq
        ctrs = lnfreq["x"]
        xmin = ctrs.min()
        xmax = ctrs.max()

        x_pred = np.linspace(xmin, xmax, num=101)  # matches R's curve
        y_pred = nlsLL(x_pred, coefs)

        if ax is None:
            ax = plt.gca()
        # Plot data
        ax.scatter(x="x", y="lnfreq", data=lnfreq,
                   alpha=0.5, label="histogram")
        # Plot predicted
        ax.plot(x_pred, y_pred, alpha=0.5, label="model")
        # Plot BEC (note this plots all BECs in becx)
        bec_x = self.bec(coefs)  # need an array for nlsLL
        bec_y = nlsLL(bec_x, coefs)
        _plot_bec(bec_x, bec_y, ax=ax, xytext=(5, 5))

        ax.legend(loc=8, bbox_to_anchor=(0.5, 1), frameon=False,
                  borderaxespad=0.1, ncol=2)
        ax.set_xlabel("x")
        ax.set_ylabel("log frequency")

        return(ax)

    def _plot_ecdf(x_pred_expm1, y_pred, ax):
        """Plot Empirical Frequency Distribution

        Plot the ECDF at predicted x and corresponding y locations.

        Parameters
        ----------
        x_pred : ndarray, shape (n,)
            Values of the variable at which to plot the ECDF.
        y_pred : ndarray, shape (n,)
            Values of the ECDF at `x_pred`.
        ax : matplotlib.Axes
            An Axes instance to use as target.

        """
        pass
