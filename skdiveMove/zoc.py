"""Class performing zero-offset correction of depth

"""

import logging
import pandas as pd
from skdiveMove.tdrsource import TDRSource
from skdiveMove.core import robjs, cv, pandas2ri
from skdiveMove.helpers import _add_xr_attr

logger = logging.getLogger(__name__)
# Add the null handler if importing as library; whatever using this library
# should set up logging.basicConfig() as needed
logger.addHandler(logging.NullHandler())


class ZOC(TDRSource):
    """Perform zero offset correction

    See help(ZOC) for inherited attributes.

    Attributes
    ----------
    zoc_params
    depth_zoc

    zoc_method : str
        Name of the ZOC method used.
    zoc_filters : pandas.DataFrame
        DataFrame with output filters for method="filter"

    """

    def __init__(self, *args, **kwargs):
        """Initialize ZOC instance

        Parameters
        ----------
        *args : positional arguments
            Passed to :meth:`TDRSource.__init__`
        **kwargs : keyword arguments
            Passed to :meth:`TDRSource.__init__`

        """
        TDRSource.__init__(self, *args, **kwargs)
        self.zoc_method = None
        self._zoc_params = None
        self._depth_zoc = None
        self.zoc_filters = None

    def __str__(self):
        base = TDRSource.__str__(self)
        meth, params = self.zoc_params
        return(base +
               ("\n{0:<20} {1}\n{2:<20} {3}"
                .format("ZOC method:", meth, "ZOC parameters:", params)))

    def _offset_depth(self, offset=0):
        """Perform ZOC with "offset" method

        Parameters
        ----------
        offset : float, optional
            Value to subtract from measured depth.

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        """
        # Retrieve copy of depth from our own property
        depth = self.depth
        self.zoc_method = "offset"
        self._zoc_params = dict(offset=offset)

        depth_zoc = depth - offset
        depth_zoc[depth_zoc < 0] = 0

        _add_xr_attr(depth_zoc, "history", "ZOC")

        self._depth_zoc = depth_zoc

    def _filter_depth(self, k, probs, depth_bounds=None, na_rm=True):
        """Perform ZOC with "filter" method

        Parameters
        ----------
        k : array_like
        probs : array_like
        **kwargs : optional keyword arguments
            For this method: ('depth_bounds' (defaults to range), 'na_rm'
            (defaults to True)).

        Notes
        -----
        More details in diveMove's ``calibrateDepth`` function.

        """
        self.zoc_method = "filter"
        # Retrieve copy of depth from our own property
        depth = self.depth

        depth_ser = depth.to_series()
        self._zoc_params = dict(k=k, probs=probs, depth_bounds=depth_bounds,
                                na_rm=na_rm)
        depthmtx = self._depth_filter_r(depth_ser, **self._zoc_params)
        depth_zoc = depthmtx.pop("depth_adj")
        depth_zoc[depth_zoc < 0] = 0
        depth_zoc = depth_zoc.rename("depth").to_xarray()
        depth_zoc.attrs = depth.attrs
        _add_xr_attr(depth_zoc, "history", "ZOC")
        self._depth_zoc = depth_zoc
        self.zoc_filters = depthmtx

    def zoc(self, method="filter", **kwargs):
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

        Examples
        --------
        ZOC using the "offset" method

        >>> from skdiveMove.tests import diveMove2skd
        >>> tdrX = diveMove2skd()
        >>> tdrX.zoc("offset", offset=3)

        Using the "filter" method

        >>> # Window lengths and probabilities
        >>> DB = [-2, 5]
        >>> K = [3, 5760]
        >>> P = [0.5, 0.02]
        >>> tdrX.zoc(k=K, probs=P, depth_bounds=DB)

        Plot the filters that were applied

        >>> tdrX.plot_zoc(ylim=[-1, 10])  # doctest: +ELLIPSIS
        (<Figure ... with 3 Axes>, array([<AxesSubplot:...>,
            <AxesSubplot:...>, <AxesSubplot:...>], dtype=object))

        """
        if method == "offset":
            offset = kwargs.pop("offset", 0)
            self._offset_depth(offset)
        elif method == "filter":
            k = kwargs.pop("k")         # must exist
            P = kwargs.pop("probs")  # must exist
            # Default depth bounds equal measured depth range
            DB = kwargs.pop("depth_bounds",
                            [self.depth.min(),
                             self.depth.max()])
            # default as in `_depth_filter`
            na_rm = kwargs.pop("na_rm", True)
            self._filter_depth(k=k, probs=P, depth_bounds=DB, na_rm=na_rm)
        else:
            logger.warning("Method {} is not implemented"
                           .format(method))

        logger.info("Finished ZOC")

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

    depth_zoc = property(_get_depth)
    """Depth array accessor

    Returns
    -------
    xarray.DataArray

    """

    def _get_params(self):
        return((self.zoc_method, self._zoc_params))

    zoc_params = property(_get_params)
    """Parameters used with method for zero-offset correction

    Returns
    -------
    method : str
        Method used for ZOC.
    params : dict
        Dictionary with parameters and values used for ZOC.

    """
