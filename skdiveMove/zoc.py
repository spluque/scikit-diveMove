"""Class performing zero-offset correction of depth

Class and Methods Summary
-------------------------

.. autosummary::

   ZOC.offset_depth
   ZOC.filter_depth
   ZOC.__call__

"""

import logging
import pandas as pd
from skdiveMove.core import robjs, cv, pandas2ri
from skdiveMove.helpers import _add_xr_attr

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


class ZOC:
    """Perform zero offset correction

    See help(ZOC) for attributes.

    """

    def __init__(self, method=None, params=None,
                 depth_zoc=None, filters=None):
        """Initialize object

        Parameters
        ----------
        method : str
            Name of the ZOC method used.
        params : dict
            See help(ZOC.params)
        depth_zoc : xarray.DataArray
            See help(ZOC.depth)
        filters : pandas.DataFrame
            DataFrame with output filters for method="filter"

        """
        self.method = method
        self._params = params
        self._depth_zoc = depth_zoc
        self.filters = filters

    def offset_depth(self, depth, offset=0):
        """Perform ZOC with "offset" method

        Parameters
        ----------
        depth : xarray.DataArray
            DataArray with observed depth measurements.
        **kwargs : optional keyword arguments
            For this method: 'offset': 0 (default).

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        """
        self.method = "offset"
        self._params = dict(offset=offset)

        depth_zoc = depth - offset
        depth_zoc[depth_zoc < 0] = 0

        _add_xr_attr(depth_zoc, "history", "ZOC")

        self._depth_zoc = depth_zoc

    def filter_depth(self, depth, k, probs, depth_bounds=None, na_rm=True):
        """Perform ZOC with "filter" method

        Parameters
        ----------
        depth : xarray.DataArray
            DataArray with observed depth measurements.
        **kwargs : optional keyword arguments
            'filter': ('k', 'probs', 'depth_bounds' (defaults to
            range), 'na_rm' (defaults to True)).

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        """
        self.method = "filter"

        depth_ser = depth.to_series()
        self._params = dict(k=k, probs=probs, depth_bounds=depth_bounds,
                            na_rm=na_rm)
        depthmtx = self._depth_filter_r(depth_ser, **self._params)
        depth_zoc = depthmtx.pop("depth_adj")
        depth_zoc[depth_zoc < 0] = 0
        depth_zoc = depth_zoc.rename("depth").to_xarray()
        depth_zoc.attrs = depth.attrs
        _add_xr_attr(depth_zoc, "history", "ZOC")
        self._depth_zoc = depth_zoc
        self.filters = depthmtx

    def __call__(self, depth, method="filter", **kwargs):
        """Apply zero offset correction to depth measurements

        Parameters
        ----------
        method : {"filter", "offset"}
            Name of method to use for zero offset correction.
        **kwargs : optional keyword arguments
            Passed to the chosen method (:meth:`offset_depth`,
            :meth:`filter_depth`)

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        """
        if method == "offset":
            offset = kwargs.pop("offset", 0)
            self.offset_depth(depth, offset)
        elif method == "filter":
            k = kwargs.pop("k")         # must exist
            P = kwargs.pop("probs")  # must exist
            # Default depth bounds equal measured depth range
            DB = kwargs.pop("depth_bounds", [depth.min(), depth.max()])
            # default as in `_depth_filter`
            na_rm = kwargs.pop("na_rm", True)
            self.filter_depth(depth, k=k, probs=P, depth_bounds=DB,
                              na_rm=na_rm)
        else:
            logger.warning("Method {} is not implemented"
                           .format(method))

    def _depth_filter_r(self, depth, k, probs, depth_bounds, na_rm=True):
        """Filter method for zero offset correction via `diveMove`

        Parameters
        ----------
        depth : pandas.Series
        k : array_like
        probs : array_like
        depth_bounds : array_like
        na_rm : bool, optional

        Returns
        -------
        out : pandas.DataFrame
            Time-indexed DataFrame with a column for each filter applied, and a
            column `depth_adj` for corrected depth.

        """
        filterFun = robjs.r("""filterFun <- diveMove:::.depthFilter""")
        with cv.localconverter(robjs.default_converter +
                               pandas2ri.converter):
            depthmtx = filterFun(depth, pd.Series(k), pd.Series(probs),
                                 pd.Series(depth_bounds), na_rm)

        colnames = ["k{0}_p{1}".format(k, p) for k, p in zip(k, probs)]
        colnames.append("depth_adj")
        return(pd.DataFrame(depthmtx, index=depth.index, columns=colnames))

    def _get_depth(self):
        return(self._depth_zoc)

    depth = property(_get_depth)
    """Depth array accessor

    Returns
    -------
    xarray.DataArray

    """

    def _get_params(self):
        return((self.method, self._params))

    params = property(_get_params)
    """Parameters used with method for zero-offset correction

    Returns
    -------
    method : str
        Method used for ZOC.
    params : dict
        Dictionary with parameters and values used for ZOC.

    """
