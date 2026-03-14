"""Utility functions for fitting an ellipsoid to IMU data

"""

import logging
import numpy as np
from scipy.linalg import sqrtm

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())

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


# Different approach from Li and Griffiths (2004), implemented in
# https://github.com/nliaudat/magnetometer_calibration

class MagnetometerCalibrator:
    """Magnetometer calibration process and application

    Attributes
    ----------
    F : float
        Presumed strength of magnetic field (uT) for the calibration.
    b : array_like (3,1)
        Hard iron bias column vector.
    A_1 : array_like (3,3)
        Soft iron transformation matrix.

    References
    ----------
    .. [1] Qingde Li; Griffiths, J.G., "Least squares ellipsoid specific
        fitting," in Geometric Modeling and Processing, 2004.  Proceedings,
        vol., no., pp.335-340, 2004

    """

    def __init__(self, field_strength):
        """Initialize calibrator

        Parameters
        ----------
        field_strength : float
            Strength of magnetic field (uT) where data were collected.

        """
        self.F = field_strength
        self.b = np.zeros([3, 1])
        self.A_1 = np.eye(3)

    def __str__(self):
        objcls = "Class {} object\n".format(self.__class__.__name__)
        F_str = ("{0} {1}\n"
                 .format("Local magnetic field strength [uT]:",
                         np.round(self.F, 2)))
        b_str = ("{0}\n{1}\n"
                 .format("Hard iron bias [uT]:",
                         np.round(self.b.flatten(), 2)))
        A_1_str = ("{0}\n{1}\n"
                   .format("Soft iron transformation matrix:",
                           np.round(self.A_1, 2)))

        return objcls + F_str + b_str + A_1_str

    def configure(self, vectors):
        """Retrieve hard iron bias and soft iron transformation matrix

        Parameters
        ----------
        vectors: (N,3) array
            Array of measured x, y, z vector components. Ideally, these data
            must have been collected following a calibration protocol.

        """
        M, n, d = self._ellipsoid_fit(vectors)
        M_1 = np.linalg.inv(M)
        self.b = -np.dot(M_1, n)
        self.A_1 = np.real(self.F /
                           np.sqrt(np.dot(n.T, np.dot(M_1, n)) - d) *
                           sqrtm(M))

    def calibrate(self, vectors):
        """Calibrate magnetometer measurements

        Parameters
        ----------
        vectors: (N,3) array
            Array of measured x, y, z vector components. Ideally, these data
            must have been collected following a calibration protocol.

        Returns:
            numpy array of calibrated data
        """
        # Subtract hard iron bias and apply soft iron correction
        vectors_calibrated = ((vectors - np.transpose(self.b)) @
                              np.transpose(self.A_1))

        return vectors_calibrated

    def _ellipsoid_fit(self, vectors):
        """Estimate ellipsoid parameters from a set of points

        Parameters
        ----------
        vectors: (N,3) array
            Array of measured x, y, z vector components. Ideally, these data
            must have been collected following a calibration protocol.

        Returns
        -------
        M, n, d : array_like, array_like, float
          The ellipsoid parameters M, n, d.

        """
        s = np.transpose(vectors)
        # D (samples)
        D = np.array([s[0] ** 2, s[1] ** 2, s[2] ** 2,
                      2 * s[1] * s[2], 2 * s[0] * s[2], 2 * s[0] * s[1],
                      2 * s[0], 2 * s[1], 2 * s[2], np.ones_like(s[0])])

        # S, S_11, S_12, S_21, S_22 (eq. 11)
        S = np.dot(D, D.T)
        S_11 = S[:6, :6]
        S_12 = S[:6, 6:]
        S_21 = S[6:, :6]
        S_22 = S[6:, 6:]

        # C (Eq. 8, k=4)
        C = np.array([[-1, 1, 1, 0, 0, 0],
                      [1, -1, 1, 0, 0, 0],
                      [1, 1, -1, 0, 0, 0],
                      [0, 0, 0, -4, 0, 0],
                      [0, 0, 0, 0, -4, 0],
                      [0, 0, 0, 0, 0, -4]])

        # v_1 (eq. 15, solution)
        E = np.dot(np.linalg.inv(C),
                   S_11 - np.dot(S_12, np.dot(np.linalg.inv(S_22), S_21)))

        E_w, E_v = np.linalg.eig(E)
        v_1 = E_v[:, np.argmax(E_w)]
        if v_1[0] < 0:
            v_1 = -v_1

        # v_2 (eq. 13, solution)
        v_2 = np.dot(np.dot(-np.linalg.inv(S_22), S_21), v_1)

        # quadratic-form parameters, parameters h and f swapped as per
        # correction by Robert R on Teslabs page
        M = np.array([[v_1[0], v_1[5], v_1[4]],
                      [v_1[5], v_1[1], v_1[3]],
                      [v_1[4], v_1[3], v_1[2]]])
        n = np.array([[v_2[0]],
                      [v_2[1]],
                      [v_2[2]]])
        d = v_2[3]

        return M, n, d
