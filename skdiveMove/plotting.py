"""Plotting module

These are considered low-level functions that do not handle the
higher-level classes of the package.

"""

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()


def _night(times, sunrise_time, sunset_time):
    """Construct Series with sunset and sunrise times for given dates

    Parameters
    ----------
    times : pandas.Series
        (N,) array with depth measurements.
    sunrise_time : str
    sunset_time : str

    Returns
    -------
    tuple
        Two pandas.Series (sunsets, sunrises)

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


def plot_tdr(depth, concur_vars=None, xlim=None, depth_lim=None,
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
        Minimum and maximum limits for ``x`` axis.  Ignored when
        ``concur_vars=None``.
    ylim : 2-tuple/list, optional
        Minimum and maximum limits for ``y`` axis for data other than depth.
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

    Returns
    -------
    tuple
        Pyplot Figure and Axes instances.

    """
    sunsets, sunrises = _night(depth.index,
                               sunset_time=sunset_time,
                               sunrise_time=sunrise_time)

    def _plot_phase_cat(ser, ax, legend=True):
        """Scatter plot and legend of series coloured by categories"""
        cats = phase_cat.cat.categories
        cat_codes = phase_cat.cat.codes
        isna_ser = ser.isna()
        ser_nona = ser.dropna()
        scatter = ax.scatter(ser_nona.index, ser_nona, s=12, marker="o",
                             c=cat_codes[~isna_ser])
        if legend:
            handles, _ = scatter.legend_elements()
            ax.legend(handles, cats, loc="lower right",
                      ncol=len(cat_codes))

    if concur_vars is None:
        fig, axs = plt.subplots(1, 1)
        axs.set_ylabel(ylab_depth)
        depth.plot(ax=axs, color="k", **kwargs)
        axs.set_xlabel("")
        axs.axhline(0, linestyle="--", linewidth=0.75, color="k")
        for beg, end in zip(sunsets, sunrises):
            axs.axvspan(beg, end, facecolor=night_col,
                        edgecolor=None, alpha=0.3)
        if (phase_cat is not None):
            _plot_phase_cat(depth, axs)
        if (dry_time is not None):
            _plot_dry_time(dry_time, axs)
        if (depth_lim is not None):
            axs.set_ylim(depth_lim)
        axs.invert_yaxis()
    else:
        full_df = pd.concat((depth, concur_vars), axis=1)
        nplots = full_df.shape[1]
        depth_ser = full_df.iloc[:, 0]
        concur_df = full_df.iloc[:, 1:]
        fig, axs = plt.subplots(nplots, 1, sharex=True)
        axs[0].set_ylabel(ylab_depth)
        depth_ser.plot(ax=axs[0], color="k", **kwargs)
        axs[0].set_xlabel("")
        axs[0].axhline(0, linestyle="--", linewidth=0.75, color="k")
        concur_df.plot(ax=axs[1:], subplots=True, legend=False, **kwargs)
        for i, col in enumerate(concur_df.columns):
            if (concur_var_titles is not None):
                axs[i + 1].set_ylabel(concur_var_titles[i])
            else:
                axs[i + 1].set_ylabel(col)
            axs[i + 1].axhline(0, linestyle="--",
                               linewidth=0.75, color="k")
            if (xlim is not None):
                axs[i + 1].set_xlim(xlim)

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

        if (depth_lim is not None):
            axs[0].set_ylim(depth_lim)

        axs[0].invert_yaxis()

    fig.tight_layout()

    return(fig, axs)


def _plot_zoc_filters(depth, zoc_filters, xlim=None, ylim=None,
                      ylab="Depth [m]", **kwargs):
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
    **kwargs : optional keyword arguments
        Passed to `matplotlib.pyplot.subplots`.  It can be any keyword,
        except for `sharex` or `sharey`.

    Returns
    -------
    tuple
        Pyplot Figure and Axes instances.

    """
    nfilters = zoc_filters.shape[1]
    npanels = 3
    lastflts = [1]              # col idx of second filters
    if nfilters > 2:            # append col idx of last filter
        lastflts.append(nfilters - 1)

    fig, axs = plt.subplots(npanels, 1, sharex=True, sharey=True, **kwargs)
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
    filter_names = zoc_filters.columns
    (zoc_filters.iloc[:, 0]
     .plot(ax=axs[1], label=filter_names[0]))  # first filter
    for i in lastflts:
        zoc_filters.iloc[:, i].plot(ax=axs[1], label=filter_names[i])
    axs[1].legend(loc="lower left")

    # ZOC depth
    depth_zoc = depth - zoc_filters.iloc[:, -1]
    depth_zoc_label = ("input - {}"
                       .format(zoc_filters.columns[-1]))
    (depth_zoc
     .plot(ax=axs[2], color="k", rot=0, label=depth_zoc_label))
    axs[2].legend(loc="lower left")
    axs[2].set_xlabel("")
    fig.tight_layout()

    return(fig, axs)


def plot_dive_model(x, depth_s, depth_deriv, d_crit, a_crit,
                    d_crit_rate, a_crit_rate, leg_title=None, **kwargs):
    """Plot dive model

    Parameters
    ----------
    x : pandas.Series
      Time-indexed depth measurements.
    depth_s : pandas.Series
      Time-indexed smoothed depth.
    depth_deriv : pandas.Series
      Time-indexed derivative of depth smoothing spline.
    d_crit : int
      Integer denoting the index where the descent ends in the observed
      time series.
    a_crit : int
      Integer denoting the index where the ascent begins in the observed
      time series.
    d_crit_rate : float
      Vertical rate of descent corresponding to the quantile used.
    a_crit_rate :
      Vertical rate of ascent corresponding to the quantile used.
    leg_title : str, optional
      Title for the plot legend (e.g. dive number being plotted).
    **kwargs : optional keyword arguments
        Passed to `matplotlib.pyplot.subplots`.  It can be any keyword,
        except `sharex`.

    Returns
    -------
    tuple
        Pyplot Figure and Axes instances.

    Notes
    -----
    The function is homologous to diveMove's `plotDiveModel`.

    """
    d_crit_time = x.index[d_crit]
    a_crit_time = x.index[a_crit]
    fig, axs = plt.subplots(2, 1, sharex=True, **kwargs)
    ax1, ax2 = axs
    ax1.invert_yaxis()
    ax1.set_ylabel("Depth")
    ax2.set_ylabel("First derivative")

    ax1.plot(x, marker="o", linewidth=0.7, color="k", label="input")
    ax1.plot(depth_s, "--", label="smooth")
    ax1.plot(x.iloc[:d_crit + 1], color="C1", label="descent")
    ax1.plot(x.iloc[a_crit:], color="C2", label="ascent")
    ax1.legend(loc="upper center", title=leg_title, ncol=2)

    ax2.plot(depth_deriv, linewidth=0.5, color="k")  # derivative
    dstyle = dict(marker=".", linestyle="None")
    ax2.plot(depth_deriv[depth_deriv > d_crit_rate].loc[:d_crit_time],
             color="C1", **dstyle)  # descent
    ax2.plot(depth_deriv[depth_deriv < a_crit_rate].loc[a_crit_time:],
             color="C2", **dstyle)  # ascent
    qstyle = dict(linestyle="--", linewidth=0.5, color="k")
    ax2.axhline(d_crit_rate, **qstyle)
    ax2.axhline(a_crit_rate, **qstyle)
    ax2.axvline(d_crit_time, **qstyle)
    ax2.axvline(a_crit_time, **qstyle)
    # Text annotation
    qiter = zip(x.index[[0, 0]],
                [d_crit_rate, a_crit_rate],
                [r"descent $\hat{q}$", r"ascent $\hat{q}$"],
                ["bottom", "top"])
    for xpos, qval, txt, valign in qiter:
        ax2.text(xpos, qval, txt, va=valign)

    titer = zip([d_crit_time, a_crit_time], [0, 0],
                ["descent", "ascent"],
                ["right", "left"])
    for ttime, ypos, txt, halign in titer:
        ax2.text(ttime, ypos, txt, ha=halign)

    return(fig, (ax1, ax2))


if __name__ == '__main__':
    from .tdr import get_diveMove_sample_data
    tdrX = get_diveMove_sample_data()
    print(tdrX)
