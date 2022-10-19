"""Utility functions for fitting an ellipsoid to IMU data

"""

import numpy as np

# Types of ellipsoid accepted fits
_ELLIPSOID_FTYPES = ["rxyz", "xyz", "xy", "xz", "yz", "sxyz"]


def fit_ellipsoid(vectors, f="rxyz"):
    """Fit a (non) rotated ellipsoid or sphere to 3D vector data

    Parameters
    ----------
    vectors: (N,3) array
        Array of measured x, y, z vector components.
    f: str
        String indicating the model to fit (one of 'rxyz', 'xyz', 'xy',
        'xz', 'yz', or 'sxyz'):
        rxyz : rotated ellipsoid (any axes)
        xyz  : non-rotated ellipsoid
        xy   : radius x=y
        xz   : radius x=z
        yz   : radius y=z
        sxyz : radius x=y=z sphere

    Returns
    -------
    otuple: tuple
        Tuple with offset, gain, and rotation matrix, in that order.

    """

    if f not in _ELLIPSOID_FTYPES:
        raise ValueError("f must be one of: {}"
                         .format(_ELLIPSOID_FTYPES))

    x = vectors[:, 0, np.newaxis]
    y = vectors[:, 1, np.newaxis]
    z = vectors[:, 2, np.newaxis]

    if f == "rxyz":
        D = np.hstack((x ** 2, y ** 2, z ** 2,
                       2 * x * y, 2 * x * z, 2 * y * z,
                       2 * x, 2 * y, 2 * z))
    elif f == "xyz":
        D = np.hstack((x ** 2, y ** 2, z ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "xy":
        D = np.hstack((x ** 2 + y ** 2, z ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "xz":
        D = np.hstack((x ** 2 + z ** 2, y ** 2,
                       2 * x, 2 * y, 2 * z))
    elif f == "yz":
        D = np.hstack((y ** 2 + z ** 2, x ** 2,
                       2 * x, 2 * y, 2 * z))
    else:                       # sxyz
        D = np.hstack((x ** 2 + y ** 2 + z ** 2,
                       2 * x, 2 * y, 2 * z))

    v = np.linalg.lstsq(D, np.ones(D.shape[0]), rcond=None)[0]

    if f == "rxyz":
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8], -1]])
        ofs = np.linalg.lstsq(-A[:3, :3], v[[6, 7, 8]], rcond=None)[0]
        Tmtx = np.eye(4)
        Tmtx[3, :3] = ofs
        AT = Tmtx @ A @ Tmtx.T    # ellipsoid translated to 0, 0, 0
        ev, rotM = np.linalg.eig(AT[:3, :3] / -AT[3, 3])
        rotM = np.fliplr(rotM)
        ev = np.flip(ev)
        gain = np.sqrt(1.0 / ev)
    else:
        if f == "xyz":
            v = np.array([v[0], v[1], v[2], 0, 0, 0, v[3], v[4], v[5]])
        elif f == "xy":
            v = np.array([v[0], v[0], v[1], 0, 0, 0, v[2], v[3], v[4]])
        elif f == "xz":
            v = np.array([v[0], v[1], v[0], 0, 0, 0, v[2], v[3], v[4]])
        elif f == "yz":
            v = np.array([v[1], v[0], v[0], 0, 0, 0, v[2], v[3], v[4]])
        else:
            v = np.array([v[0], v[0], v[0], 0, 0, 0, v[1], v[2], v[3]])

        ofs = -(v[6:] / v[:3])
        rotM = np.eye(3)
        g = 1 + (v[6] ** 2 / v[0] + v[7] ** 2 / v[1] + v[8] ** 2 / v[2])
        gain = (np.sqrt(g / v[:3]))

    return (ofs, gain, rotM)


def _refine_ellipsoid_fit(gain, rotM):
    """Refine ellipsoid fit"""
    # m = 0
    # rm = 0
    # cm = 0
    pass


def apply_ellipsoid(vectors, offset, gain, rotM, ref_r):
    """Apply ellipsoid fit to vector array"""
    vectors_new = vectors.copy() - offset
    vectors_new = vectors_new @ rotM
    # Scale to sphere
    vectors_new = vectors_new / gain * ref_r
    return vectors_new
