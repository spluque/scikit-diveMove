"""Speed calibration algorithm

This is a reimplementation of the algorithm in diveMove, as it was too
cumbersome to use rpy2 for this operation.

"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


def calibrate_speed(x, tau, contour_level, z=0, bad=[0, 0],
                    plot=True, ax=None):
    """Calibration based on kernel density estimation

    Parameters
    ----------
    x : pandas.DataFrame
        DataFrame with depth rate and speed
    tau : float
    contour_level : float
    z : float, optional
    bad : array_like, optional
    plot : bool, optional
        Whether to plot calibration results.
    ax : matplotlib.Axes, optional
        An Axes instance to use as target.  Default is to create one.

    Returns
    -------
    out : 2-tuple
        The quantile regression fit object, and `matplotlib.pyplot` `Axes`
        instance (if plot=True, otherwise None).

    Notes
    -----
    See `skdiveMove.TDR.calibrate_speed` for details.

    """
    # `gaussian_kde` expects variables in rows
    n_eval = 51

    # Numpy for some operations
    xnpy = x.to_numpy()

    kde = stats.gaussian_kde(xnpy.T)
    # Build the grid for evaluation, mimicking bkde2D
    mins = x.min()
    maxs = x.max()
    x_flat = np.linspace(mins[0], maxs[0], n_eval)
    y_flat = np.linspace(mins[1], maxs[1], n_eval)
    xx, yy = np.meshgrid(x_flat, y_flat)
    grid_coords = np.append(xx.reshape(-1, 1), yy.reshape(-1, 1), axis=1)
    # Evaluate kde on the grid
    z = kde(grid_coords.T)
    z = np.flipud(z.reshape(n_eval, n_eval))

    # Fit quantile regression
    # -----------------------
    # Bin depth rate
    drbinned = pd.cut(x.iloc[:, 0], n_eval)
    drbin_mids = drbinned.apply(lambda x: x.mid)  # mid points
    # Use bin mid points as x
    binned = np.column_stack((drbin_mids, xnpy[:, 1]))
    qdata = pd.DataFrame(binned, columns=list("xy"))
    qmod = smf.quantreg("y ~ x", qdata)
    qfit = qmod.fit(q=tau)
    coefs = qfit.params
    logger.info("a={}, b={}".format(*coefs))

    if plot:
        fig = plt.gcf()
        if ax is None:
            ax = plt.gca()

        ax.set_xlabel("Rate of depth change")
        ax.set_ylabel("Speed")
        zimg = ax.imshow(z, aspect="auto",
                         extent=[mins[0], maxs[0], mins[1], maxs[1]],
                         cmap="gist_earth_r")
        fig.colorbar(zimg, fraction=0.1, aspect=30, pad=0.02,
                     label="Kernel density probability")
        cntr = ax.contour(z, extent=[mins[0], maxs[0], mins[1], maxs[1]],
                          origin="image", levels=[contour_level])
        ax.clabel(cntr, fmt="%1.2f")
        # Plot the binned data, adding some noise for clarity
        xjit_binned = np.random.normal(binned[:, 0],
                                       xnpy[:, 0].ptp() / (2 * n_eval))
        ax.scatter(xjit_binned, binned[:, 1], s=6, alpha=0.3)
        # Plot line
        xnew = np.linspace(mins[0], maxs[0])
        yhat = coefs[0] + coefs[1] * xnew
        ax.plot(xnew, yhat, "--k",
                label=(r"$y = {:.3f} {:+.3f} x$"
                       .format(coefs[0], coefs[1])))
        ax.legend(loc="lower right")
        # Adjust limits to compensate for the noise in x
        ax.set_xlim([mins[0], maxs[0]])

    return(qfit, ax)


if __name__ == '__main__':
    from skdiveMove.tests import diveMove2skd
    tdrX = diveMove2skd()
    print(tdrX)
