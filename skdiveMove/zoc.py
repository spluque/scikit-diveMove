"""Class performing zero-offset correction of depth

"""

import logging
import pandas as pd
from skdiveMove.tdrsource import TDRSource
from skdiveMove.core import robjs, pandas2ri, diveMove
from skdiveMove.helpers import _append_xr_attr

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
        *args
            Positional arguments passed to :meth:`TDRSource.__init__`
        **kwargs
            Keyword arguments passed to :meth:`TDRSource.__init__`

        """
        TDRSource.__init__(self, *args, **kwargs)
        self.zoc_method = None
        self._zoc_params = None
        self._depth_zoc = None
        self.zoc_filters = None

    def __str__(self):
        base = TDRSource.__str__(self)
        meth, params = self.zoc_params
        return (base +
                ("\n{0:<20} {1}\n{2:<20} {3}"
                 .format("ZOC method:", meth, "ZOC parameters:", params)))

    def _offset_depth(self, offset=0):
        """Perform ZOC with "offset" method

        Parameters
        ----------
        offset : float, optional
            Value to subtract from measured depth.

        """
        # Retrieve copy of depth from our own property
        depth = self.depth
        self.zoc_method = "offset"
        self._zoc_params = dict(offset=offset)

        depth_zoc = depth - offset
        depth_zoc[depth_zoc < 0] = 0

        _append_xr_attr(depth_zoc, "history", "ZOC")

        self._depth_zoc = depth_zoc

    def _filter_depth(self, k, probs, depth_bounds=None, na_rm=True):
        """Perform ZOC with "filter" method

        Parameters
        ----------
        k : array_like
        probs : array_like
        **kwargs
            Optional keyword arguments. For this method: ('depth_bounds'
            (defaults to range), 'na_rm' (defaults to True)).

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
        _append_xr_attr(depth_zoc, "history", "ZOC")
        self._depth_zoc = depth_zoc
        self.zoc_filters = depthmtx

    def zoc(self, method="filter", **kwargs):
        """Apply zero offset correction to depth measurements

        This procedure is required to correct drifts in the pressure
        transducer of TDR records and noise in depth measurements. Three
        methods are available to perform this correction.

        Method `offset` can be used when the offset is known in advance,
        and this value is used to correct the entire time series.
        Therefore, ``offset=0`` specifies no correction.

        Method `filter` implements a smoothing/filtering mechanism where
        running quantiles can be applied to depth measurements in a
        recursive manner [3]_.  The method calculates the first running
        quantile defined by the first probability in a given sequence on a
        moving window of size specified by the first integer supplied in a
        second sequence.  The next running quantile, defined by the second
        supplied probability and moving window size, is applied to the
        smoothed/filtered depth measurements from the previous step, and so
        on. The corrected depth measurements (d) are calculated as:

        .. math::

           d = d_{0} - d_{n}

        where :math:`d_{0}` is original depth and :math:`d_{n}` is the last
        smoothed/filtered depth.  This method is under development, but
        reasonable results can be achieved by applying two filters (see
        Examples). The default `na_rm=True` works well when there are no
        level shifts between non-NA phases in the data, but `na_rm=False`
        is better in the presence of such shifts. In other words, there is
        no reason to pollute the moving window with null values when
        non-null phases can be regarded as a continuum, so splicing
        non-null phases makes sense. Conversely, if there are level shifts
        between non-null phases, then it is better to retain null phases to
        help the algorithm recognize the shifts while sliding the
        window(s). The search for the surface can be limited to specified
        bounds during smoothing/filtering, so that observations outside
        these bounds are interpolated using the bounded smoothed/filtered
        series.

        Once the entire record has been zero-offset corrected, remaining
        depths below zero, are set to zero, as these are assumed to
        indicate values at the surface.

        Parameters
        ----------
        method : {"filter", "offset"}
            Name of method to use for zero offset correction.
        **kwargs
            Optional keyword arguments passed to the chosen method
            (:meth:`_offset_depth`, :meth:`_filter_depth`)

        References
        ----------

        .. [3] Luque, S.P. and Fried, R. (2011) Recursive filtering for
           zero offset correction of diving depth time series. PLoS ONE
           6:e15850.

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
        (<Figure ... with 3 Axes>, array([<Axes: ...>,
            <Axes: ...>, <Axes: ...>], dtype=object))

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
        with (robjs.default_converter + pandas2ri.converter).context():
            depthmtx = diveMove._depthFilter(depth,
                                             pd.Series(k), pd.Series(probs),
                                             pd.Series(depth_bounds),
                                             na_rm)

        colnames = ["k{0}_p{1}".format(k, p) for k, p in zip(k, probs)]
        colnames.append("depth_adj")
        return pd.DataFrame(depthmtx, index=depth.index, columns=colnames)

    def _get_depth(self):
        return self._depth_zoc

    depth_zoc = property(_get_depth)
    """Depth array accessor

    Returns
    -------
    xarray.DataArray

    """

    def _get_params(self):
        return (self.zoc_method, self._zoc_params)

    zoc_params = property(_get_params)
    """Parameters used with method for zero-offset correction

    Returns
    -------
    method : str
        Method used for ZOC.
    params : dict
        Dictionary with parameters and values used for ZOC.

    """
