"""Utility functions for Allan Deviation analysis of IMU data

"""

import numpy as np
from scipy.optimize import curve_fit

# Mapping of error type with corresponding tau and slope
_ERROR_DEFS = {"Q": [np.sqrt(3), -1], "ARW": [1.0, -0.5],
               "BI": [np.nan, 0], "RRW": [3.0, 0.5],
               "RR": [np.sqrt(2), 1]}


def _armav_nls_fun(x, *args):
    coefs = np.array(args).reshape(len(args), 1)
    return np.log10(np.dot(x, coefs ** 2)).flatten()


def _armav(taus, adevs):
    nsize = taus.size
    # Linear regressor matrix
    x0 = np.sqrt(np.column_stack([3 / (taus ** 2), 1 / taus,
                                  np.ones(nsize), taus / 3,
                                  taus ** 2 / 2]))
    # Ridge regression bias constant
    lambda0 = 5e-3
    id0 = np.eye(5)
    sigma0 = np.linalg.solve((np.dot(x0.T, x0) + lambda0 * id0),
                             np.dot(x0.T, adevs))

    # TODO: need to be able to set bounds
    popt, pcov = curve_fit(_armav_nls_fun, x0 ** 2,
                           np.log10(adevs ** 2), p0=sigma0)

    # Compute the bias instability
    sigma_hat = np.abs(popt)
    adev_reg = np.sqrt(np.dot(x0 ** 2, sigma_hat ** 2))
    sigma_hat[2] = np.min(adev_reg) / np.sqrt((2 * np.log(2) / np.pi))

    return (sigma_hat, popt, adev_reg)


def _line_fun(t, alpha, tau_crit, adev_crit):
    """Find Allan sigma coefficient from line and point

    Log-log parameterization of the point-slope line equation.

    Parameters
    ----------
    t : {float, array_like}
        Averaging time
    alpha : float
        Slope of Allan deviation line
    tau_crit : float
        Observed averaging time
    adev_crit : float
        Observed Allan deviation at `tau_crit`
    """
    return (10 ** (alpha * (np.log10(t) - np.log10(tau_crit)) +
                   np.log10(adev_crit)))


def allan_coefs(taus, adevs):
    """Compute Allan deviation coefficients for each error type

    Given averaging intervals ``taus`` and corresponding Allan deviation
    ``adevs``, compute the Allan deviation coefficient for each error type:

    - Quantization
    - (Angle, Velocity) Random Walk
    - Bias Instability
    - Rate Random Walk
    - Rate Ramp

    Parameters
    ----------
    taus : array_like
        Averaging times
    adevs : array_like
        Allan deviation

    Returns
    -------
    sigmas_hat: dict
        Dictionary with `tau` value and associated Allan deviation
        coefficient for each error type.
    adev_reg : numpy.ndarray
        The array of Allan deviations fitted to `taus`.

    """
    # Fit ARMAV model
    sigmas_hat, popt, adev_reg = _armav(taus, adevs)
    sigmas_d = dict(zip(_ERROR_DEFS.keys(), sigmas_hat))

    return (sigmas_d, adev_reg)
