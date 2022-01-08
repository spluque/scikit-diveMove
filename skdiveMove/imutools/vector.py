"""Utilities for common vector operations


"""
import numpy as np
from scipy.spatial.transform import Rotation as R


def normalize(v):
    """Normalize vector

    Parameters
    ----------
    v : array_like (N,) or (M,N)
        input vector

    Returns
    -------
    numpy.ndarray
        Normalized vector having magnitude 1.

    """
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def vangle(v1, v2):
    """Angle between one or more vectors

    Parameters
    ----------
    v1 : array_like (N,) or (M,N)
        vector 1
    v2 : array_like (N,) or (M,N)
        vector 2

    Returns
    -------
    angle : double or numpy.ndarray(M,)
        angle between v1 and v2

    Example
    -------
    >>> v1 = np.array([[1,2,3],
    ...                [4,5,6]])
    >>> v2 = np.array([[1,0,0],
    ...                [0,1,0]])
    >>> vangle(v1,v2)
    array([1.30024656, 0.96453036])

    Notes
    -----
    .. image:: .static/images/vector_angle.png
       :scale: 75%

    .. math::

       \\alpha =arccos(\\frac{\\vec{v_1} \\cdot \\vec{v_2}}{| \\vec{v_1} |
       \\cdot | \\vec{v_2}|})

    """
    v1_norm = v1 / np.linalg.norm(v1, axis=-1, keepdims=True)
    v2_norm = v2 / np.linalg.norm(v2, axis=-1, keepdims=True)
    v1v2 = np.einsum("ij,ij->i", *np.atleast_2d(v1_norm, v2_norm))
    angle = np.arccos(v1v2)

    if len(angle) == 1:
        angle = angle.item()

    return angle


def rotate_vector(vector, q, inverse=False):
    """Apply rotations to vector or array of vectors given quaternions

    Parameters
    ----------
    vector : array_like
        One (1D) or more (2D) array with vectors to rotate.
    q : array_like
        One (1D) or more (2D) array with quaternion vectors.  The scalar
        component must be last to match `scipy`'s convention.

    Returns
    -------
    numpy.ndarray
        The rotated input vector array.

    Notes
    -----
    .. image:: .static/images/vector_rotate.png
       :scale: 75%

    .. math::

       q \\circ \\left( {\\vec x \\cdot \\vec I} \\right) \\circ {q^{ - 1}} =
       \\left( {{\\bf{R}} \\cdot \\vec x} \\right) \\cdot \\vec I

    More info under
    http://en.wikipedia.org/wiki/Quaternion

    """
    rotator = R.from_quat(q)
    return rotator.apply(vector, inverse=inverse)
