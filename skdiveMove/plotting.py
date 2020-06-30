"""Plotting module

"""

import warnings
import pandas as pd
import matplotlib.pyplot as plt

_FIGSIZE = (15, 12)


def _night(times, sunrise_time, sunset_time):
    """Construct Series with sunset and sunrise times for given dates

    Parameters
    ----------
    times : pandas.Series
        (N,) array with depth measurements.
    sunrise_time : str
    sunset_time : str

    """
    tmin = times.min().strftime("%Y-%m-%d ")
    tmax = times.max().strftime("%Y-%m-%d ")
    sunsets = pd.date_range(start=tmin + sunset_time,
                            end=tmax + sunset_time,
                            freq="1D")
    tmin1 = (times.min() + pd.Timedelta(1, unit="d")).strftime("%Y-%m-%d ")
    tmax1 = (times.max() + pd.Timedelta(1, unit="d")).strftime("%Y-%m-%d ")
    sunrises = pd.date_range(start=tmin1 + sunrise_time,
                             end=tmax1 + sunrise_time,
                             freq="1D")
    return(sunsets, sunrises)


def _plot_dry_time(times_dataframe, ax):
    """Fill a vertical span between beginning/ending times in DataFrame

    Parameters
    ----------
    times_dataframe : pandas.DataFrame
    ax: Axes object

    """
    for idx, row in times_dataframe.iterrows():
        ax.axvspan(row[0], row[1], ymin=0.99, facecolor="tan",
                   edgecolor=None, alpha=0.6)


def plotTDR(depth, concur_vars=None, xlim=None, depth_lim=None,
            xlab="time [dd-mmm hh:mm]", ylab_depth="depth [m]",
            concur_var_titles=None, xlab_format="%d-%b %H:%M",
            sunrise_time="06:00:00", sunset_time="18:00:00",
            night_col="gray", dry_time=None, phase_cat=None,
            key=True, **kwargs):
    """Plot time, depth, and other concurrent data

    Parameters
    ----------
    depth : pandas.Series
        (N,) array with depth measurements.
    concur_vars : pandas.Series or pandas.Dataframe
        (N,) Series or dataframe with additional data to plot in subplot.
    xlim : 2-tuple/list, optional
        Minimum and maximum limits for ``x`` axis.
    ylim : 2-tuple/list, optional
        Minimum and maximum limits for ``y`` axis.
    depth_lim : 2-tuple/list, optional
        Minimum and maximum limits for depth to plot.
    xlab : str, optional
        Label for ``x`` axis.
    ylab_depth : str, optional
        Label for ``y`` axis for depth.
    concur_var_titles : str or list, optional
        String or list of strings with y-axis labels for `concur_vars`.
    xlab_format : str, optional
        Format string for formatting the x axis.
    sunrise_time : str, optional
        Time of sunrise, in 24 hr format.  This is used for shading night
        time.
    sunset_time : str, optional
        Time of sunset, in 24 hr format.  This is used for shading night
        time.
    night_col : str, optional
        Color for shading night time.
    dry_time : pandas.DataFrame, optional
        Two-column DataFrame with beginning and ending times corresponding
        to periods considered to be dry.
    phase_cat : pandas.Series, optional
        Categorical series dividing rows into sections.
    **kwargs : optional keyword arguments

    """
    sunsets, sunrises = _night(depth.index,
                               sunset_time=sunset_time,
                               sunrise_time=sunrise_time)

    def _plot_phase_cat(ser, ax, legend=True):
        """Scatter plot and legend of series coloured by categories"""
        cats = phase_cat.cat.categories
        cat_codes = phase_cat.cat.codes
        scatter = ax.scatter(ser.index, ser, s=12, marker="o", c=cat_codes)
        if legend:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning,
                                        module="numpy")
                warnings.filterwarnings("ignore", category=UserWarning,
                                        module="matplotlib")
                handles, _ = scatter.legend_elements()
                ax.legend(handles, cats, loc="lower right",
                          ncol=len(cat_codes))

    if concur_vars is None:
        fig, axs = plt.subplots(1, 1, figsize=_FIGSIZE)
        axs.set_ylabel(ylab_depth)
        axs.invert_yaxis()
        depth.plot(ax=axs, color="k", **kwargs)
        axs.axhline(0, linestyle="--", linewidth=0.75, color="k")
        for beg, end in zip(sunsets, sunrises):
            axs.axvspan(beg, end, facecolor=night_col,
                        edgecolor=None, alpha=0.3)
        if (phase_cat is not None):
            _plot_phase_cat(depth, axs)
        if (dry_time is not None):
            _plot_dry_time(dry_time, axs)
        if (xlim is not None):
            axs.set_xlim(xlim)
        if (depth_lim is not None):
            axs.set_ylim(depth_lim)
    else:
        full_df = pd.concat((depth, concur_vars), axis=1)
        nplots = full_df.shape[1]
        depth_ser = full_df.iloc[:, 0]
        concur_df = full_df.iloc[:, 1:]
        fig, axs = plt.subplots(nplots, 1, sharex=True, figsize=_FIGSIZE)
        axs[0].set_ylabel(ylab_depth)
        axs[0].invert_yaxis()
        depth_ser.plot(ax=axs[0], color="k", **kwargs)
        axs[0].axhline(0, linestyle="--", linewidth=0.75, color="k")
        concur_df.plot(ax=axs[1:], subplots=True, legend=False, **kwargs)
        for i, col in enumerate(concur_df.columns):
            if (concur_var_titles is not None):
                axs[i + 1].set_ylabel(concur_var_titles[i])
            else:
                axs[i + 1].set_ylabel(col)
            axs[i + 1].axhline(0, linestyle="--",
                               linewidth=0.75, color="k")
        for i, ax in enumerate(axs):
            for beg, end in zip(sunsets, sunrises):
                ax.axvspan(beg, end, facecolor=night_col,
                           edgecolor=None, alpha=0.3)
            if (dry_time is not None):
                _plot_dry_time(dry_time, ax)

        if (phase_cat is not None):
            _plot_phase_cat(depth_ser, axs[0])
            for i, col in enumerate(concur_df.columns):
                _plot_phase_cat(concur_df.loc[:, col], axs[i + 1], False)

        if (xlim is not None):
            axs[0].set_xlim(xlim)
        if (depth_lim is not None):
            axs[0].set_ylim(depth_lim)

    fig.tight_layout()

    return(fig, axs)


def _plotZOCfilters(depth, zoc_filters, xlim=None, ylim=None,
                    ylab="Depth [m]"):
    """Plot zero offset correction filters

    Parameters
    ----------
    depth : pandas.Series
        Measured depth time series, indexed by datetime.
    zoc_filters : pandas.DataFrame
        DataFrame with ZOC filters in columns.  Must have the same number
        of records as `depth`.
    xlim : 2-tuple/list
    ylim : 2-tuple/list
    ylab : str
        Label for `y` axis.

    """
    nfilters = zoc_filters.shape[1]
    npanels = 3
    lastflts = [1]
    if nfilters > 3:
        lastflts.append(nfilters - 1)

    fig, axs = plt.subplots(npanels, 1, sharex=True, sharey=True)
    if xlim:
        axs[0].set_xlim(xlim)
    else:
        depth_nona = depth.dropna()
        axs[0].set_xlim((depth_nona.index.min(),
                         depth_nona.index.max()))
    if ylim:
        axs[0].set_ylim(ylim)
    else:
        axs[0].set_ylim((depth.min(), depth.max()))

    for ax in axs:
        ax.set_ylabel(ylab)
        ax.invert_yaxis()
        ax.axhline(0, linestyle="--", linewidth=0.75, color="k")
    depth.plot(ax=axs[0], color="lightgray", label="input")
    axs[0].legend(loc="lower left")
    # Need to plot legend for input depth here
    filter_names = zoc_filters.columns[:-1]
    flt0 = (zoc_filters.iloc[:, 0]
            .plot(ax=axs[1], label=filter_names[0]))  # first filter
    flts_l = [flt0]
    if lastflts[0] < (nfilters - 1):
        for i in lastflts:
            flt_i = zoc_filters.iloc[:, i].plot(ax=axs[1])
            flts_l.append(flt_i)
    axs[1].legend(loc="lower left")

    # ZOC depth
    depth_zoc_label = ("input - {}"
                       .format(zoc_filters.columns[-2]))
    (zoc_filters.iloc[:, -1]
     .plot(ax=axs[2], color="k", rot=0, label=depth_zoc_label))
    axs[2].legend(loc="lower left")
    axs[2].set_xlabel("")
    fig.tight_layout()


def plot_dive_model(x, y, times_s, depths_s, d_crit, a_crit,
                    times_deriv, depths_deriv, d_crit_rate, a_crit_rate,
                    diveNo=1):
    """Plot dive model

    Parameters
    ----------
    x : array_like
      Array of time step observations.
    y : array_like
      Array of depth observations at each time step in `x`.
    times_s : array_like
      Array of time steps used to generate the smoothing spline
      (i.e. knots).
    depths_s : array_like
      Array with smoothed depth along `times_s`.
    d_crit : int
      Integer denoting the index where the descent ends in the observed
      time series.
    a_crit : int
      Integer denoting the index where the ascent begins in the observed
      time series.
    times_deriv : array_like
      Array with the time steps where the derivative of the smoothing
      spline was evaluated.
    depths_deriv : array_like
      Array with the derivative of the smoothing spline evaluated at
      `times_deriv`.
    d_crit_rate : float
      Vertical rate of descent corresponding to the quantile used.
    a_crit_rate :
      Vertical rate of ascent corresponding to the quantile used.
    diveNo : int, optional
      Integer for labelling the dive number being plotted.

    Notes
    -----
    The function is homologous to diveMove's plotDiveModel.

    """
    pass


if __name__ == '__main__':
    from tdr import TDR
    tdr = TDR(("/home/sluque/Scripts/R/src/diveMove/diveMove"
               "/data/dives.csv"), sep=";", compression="bz2")
    # print(tdr)
    # beg, end = _night(tdr.tdr.index, sunset_time="18:00",
    #                   sunrise_time="06:00")
    # plotTDR(tdr.tdr["depth"], tdr.tdr[["speed", "light"]], style=".-")
    # plt.show()

    tdr.zoc("offset", offset=3)
    tdr.detect_wet()
    tdr.detect_dives(3)
    tdr.detect_dive_phases("unimodal", descent_crit_q=0.01,
                           ascent_crit_q=0, knot_factor=20)
    ccdata = tdr.tdr["speed"]
    depth = tdr.get_depth("zoc")
    wet_df = tdr.get_wet_activity("phases")
    dives_detail = tdr.get_dive_details("row_ids")

    plotTDR(depth, dry_time=wet_df["phase_label"],
            phase_cat=dives_detail["dive.phase"])
