"""Speed calibration algorithm

This is a reimplementation of the algorithm in diveMove, as it was too
cumbersome to use rpy2 for this operation.

"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


def calibrate(x, tau, contour_level, z=0, bad=[0, 0], **kwargs):
    """Calibration based on kernel density estimation

    Parameters
    ----------
    x : pandas.DataFrame
        DataFrame with depth rate and speed
    tau : float
    contour_level : float
    z : float, optional
    bad : array_like, optional
    **kwargs : optional keyword arguments
        Arguments passed to ``matplotlib.pyplot.subplots``.

    Returns
    -------
    out :

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

    fig, ax = plt.subplots(1, 1, **kwargs)
    ax.set_xlabel("Rate of depth change")
    ax.set_ylabel("Speed")
    zimg = ax.imshow(z, aspect=x_flat.ptp() / y_flat.ptp(),
                     extent=[mins[0], maxs[0], mins[1], maxs[1]],
                     cmap="gist_earth_r")
    fig.colorbar(zimg, fraction=0.1, aspect=30, pad=0.02)
    cntr = ax.contour(z, extent=[mins[0], maxs[0], mins[1], maxs[1]],
                      origin="image", levels=[contour_level])
    ax.clabel(cntr, fmt="%1.2f")
    # # Equivalent to R's `contourLines`
    # cntr_vertices = cntr.collections[0].get_paths()[0].vertices
    # # Scatter with input data
    # ax.scatter(xnpy[:, 0], xnpy[:, 1], marker=".")

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
    # Plot the binned data, adding some noise for clarity
    xjit_binned = np.random.normal(binned[:, 0],
                                   xnpy[:, 0].ptp() / (2 * n_eval))
    ax.scatter(xjit_binned, binned[:, 1], s=6, alpha=0.3)
    # # Plot line (using the contour line vertices in qdata)
    # x_new = np.linspace(cntr_vertices[:, 0].min(),
    #                     cntr_vertices[:, 0].max())
    # yhat = coefs[0] + coefs[1] * x_new
    # ax.plot(x_new, yhat)
    # Plot line
    xnew = np.linspace(mins[0], maxs[0])
    yhat = coefs[0] + coefs[1] * xnew
    ax.plot(xnew, yhat, "--k",
            label=(r"$y = {:.3f} {:+.3f} x$"
                   .format(coefs[0], coefs[1])))
    ax.legend(loc="lower right")
    # Adjust limits to compensate for the noise in x
    ax.set_xlim([mins[0], maxs[0]])

    return(qfit, fig, ax)


if __name__ == '__main__':
    from .tdr import get_diveMove_sample_data
    tdrX = get_diveMove_sample_data()
    print(tdrX)