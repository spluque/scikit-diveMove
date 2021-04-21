"""scikit-diveMove tests"""

import numpy as np
from .get_sample_data import diveMove2skd  # noqa: F401


def simulate_mixexp(n, p, lda, rng=None):
    """Generate samples from mixture of exponential distributions

    Simulate a mixture of two or three random exponential distributions.
    This uses a special definition for the probabilities (:math:`p_i`).  In
    the two-process case, :math:`p` represents the proportion of "fast" to
    "slow" events in the mixture.  In the three-process case, :math:`p_0`
    represents the proportion of "fast" to "slow" events, and :math:`p_1`
    represents the proportion of "slow" to "slow" *and* "very slow" events.

    Parameters
    ----------
    n : int
        Output sample size.
    p : float or array_like
        Probabilities for processes in the output mixture sample.
    lda : array_like
        array_like with lambda (scale) for each process.
    rng : Generator
        Random number generator object.  If not provided, a default one is
        created.

    Returns
    -------
    `ndarray`

    """
    if rng is None:
        rng = np.random.default_rng()

    if np.isscalar(p):
        p_full = np.array([p, 1 - p])
    elif len(p) == 2:
        # compute slow and very slow proc props
        p0 = p[0]
        p1 = p[1] * (1 - p0)
        p2 = 1 - (p0 + p1)
        p_full = np.array([p0, p1, p2])
    else:
        msg = ("Mixtures of more than three process not yet implemented")
        raise NotImplementedError(msg)

    rng = np.random.default_rng()
    chooser = rng.choice(len(lda), size=n, replace=True,
                         p=p_full / p_full.sum())
    rates = 1 / np.array(lda)
    proc_mix = rng.exponential(rates[chooser])
    return(proc_mix)
